"""
multi_domain_architecture.py
Multi-Domain Architecture Synthesis - unified multimodal encoder and training framework.

    Part 1  Image transformation + dynamic sequence packing (custom_collate_fn)
    Part 2  Consolidated multi-domain nn.Module: frozen pretrained ResNet18 vision path,
            LSTM sequence path, and a multi-head scaled dot-product self-attention fusion
    Part 3  Rank-shielded, MLflow-instrumented training step for multi-node execution

Run on one process:          python multi_domain_architecture.py
Run on N processes (DDP):    torchrun --nproc_per_node=N multi_domain_architecture.py
(Multi-node: torchrun --nnodes=<nodes> --node_rank=<i> --master_addr=<host> ... same script.)
"""

import math
import os
import warnings

import mlflow
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import models, transforms
from torchvision.transforms import functional as TF


def get_rank() -> int:
    """Global process rank injected by the launcher (torchrun, Azure ML, SLURM). Defaults to 0."""
    return int(os.environ.get("RANK", "0"))


# --- One-time MLflow tracking setup (must run before any mlflow.log_metric call) ---
mlflow.set_tracking_uri("sqlite:////tmp/mlflow.db")
if get_rank() == 0:
    # Only the master process creates/selects the experiment, so worker processes
    # never race each other initialising the same SQLite tracking database.
    mlflow.set_experiment("multi-domain-architecture")


# =============================================================================
# PART 1: Data Preprocessing and Collation
# =============================================================================
# ImageNet normalisation so inputs match the statistics the pretrained ResNet18 was trained on.
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def custom_collate_fn(batch):
    """
    Collate a list of (image_tensor, sequence_tensor) pairs into one model-ready batch.

    Args:
        batch: list of tuples (image [3, H, W], sequence [L_i, F]); L_i varies per sample.

    Returns:
        images  : FloatTensor [B, 3, H, W], reordered with the same permutation as the text
        packed  : PackedSequence holding the padded sequences, longest first
        lengths : LongTensor [B] on CPU, exact sequence lengths in descending order
    """
    # 1. Separate the two modalities.
    images, sequences = zip(*batch)
    lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
    if (lengths == 0).any():
        raise ValueError("Zero-length sequences cannot be packed; filter them out upstream.")

    # 2. Sort by length, longest first, and apply the SAME permutation to the images
    #    so every (image, text) pair stays aligned after reordering.
    sorted_lengths, sort_idx = torch.sort(lengths, descending=True)
    images = torch.stack(images, dim=0)[sort_idx]
    sorted_sequences = [sequences[i] for i in sort_idx.tolist()]

    # 3. Pad to the longest sequence in THIS batch, then pack so the LSTM skips padded steps.
    padded = rnn_utils.pad_sequence(sorted_sequences, batch_first=True, padding_value=0.0)
    packed = rnn_utils.pack_padded_sequence(
        padded, sorted_lengths, batch_first=True, enforce_sorted=True
    )
    return images, packed, sorted_lengths


# =============================================================================
# PART 2: Consolidated Object-Oriented Core
# =============================================================================
def _load_pretrained_resnet18() -> nn.Module:
    """Pretrained ResNet18; falls back to random init only if the weights cannot be downloaded."""
    try:
        return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception as exc:  # e.g. an offline compute node with no cached weights
        warnings.warn(
            f"Pretrained ResNet18 weights unavailable ({exc}); using random initialisation.",
            RuntimeWarning,
        )
        return models.resnet18(weights=None)


class UnifiedMultiDomainModel(nn.Module):
    """
    Vision path : frozen pretrained ResNet18 -> 512-d feature -> linear projection = 1 image token
    Text path   : LSTM over the PackedSequence -> one token per real (non-padded) time step
    Fusion      : multi-head scaled dot-product self-attention over [image token ; text tokens]
                  with a key-padding mask, residual + LayerNorm, masked mean-pool, classifier
    """

    def __init__(self, hidden_dim=128, n_heads=4, seq_input_dim=32, num_classes=2):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads}).")
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.d_k = hidden_dim // n_heads  # per-head key dimension used in the 1/sqrt(d_k) scaling

        # Register a pretrained feature backbone (ResNet18) and freeze its parameters.
        self.backbone = _load_pretrained_resnet18()
        backbone_dim = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Identity()             # keep features, drop the ImageNet classifier
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.image_proj = nn.Linear(backbone_dim, hidden_dim)

        # LSTM layer to process the variable-length sequence data.
        self.lstm = nn.LSTM(input_size=seq_input_dim, hidden_size=hidden_dim,
                            num_layers=1, batch_first=True)

        # Linear projection layers for Q, K, V (+ output projection) in the attention mechanism.
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def train(self, mode: bool = True):
        """Frozen weights are not enough: also keep the backbone's BatchNorm running stats fixed."""
        super().train(mode)
        self.backbone.eval()
        return self

    def attention(self, q, k, v, key_padding_mask=None):
        """
        Multi-head scaled dot-product attention: softmax((Q @ K^T) / sqrt(d_k)) @ V

        q, k, v          : [B, T, hidden_dim] (already projected)
        key_padding_mask : optional BoolTensor [B, T], True where the position is padding
        returns          : [B, T, hidden_dim]
        """
        batch_size, num_tokens, _ = q.shape

        def split_heads(x):  # [B, T, hidden_dim] -> [B, n_heads, T, d_k]
            return x.view(batch_size, num_tokens, self.n_heads, self.d_k).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)       # [B, n_heads, T, T]
        if key_padding_mask is not None:
            # Padded keys receive zero attention weight. The image token is never masked,
            # so no row is fully masked and the softmax cannot produce NaN.
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = weights @ v                                           # [B, n_heads, T, d_k]
        return context.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.hidden_dim)

    def forward(self, image_inputs, packed_sequence_inputs):
        # Vision path: frozen backbone -> one image token.
        image_features = self.backbone(image_inputs)                    # [B, 512]
        image_token = self.image_proj(image_features).unsqueeze(1)      # [B, 1, hidden_dim]

        # Sequence path: the LSTM consumes the PackedSequence directly, so padded steps are never
        # computed; unpack afterwards to obtain per-step tokens plus their true lengths.
        packed_output, _ = self.lstm(packed_sequence_inputs)
        text_tokens, lengths = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)

        # Joint token set (both paths now share hidden_dim) and its key-padding mask.
        tokens = torch.cat([image_token, text_tokens], dim=1)           # [B, 1 + T, hidden_dim]
        max_len = text_tokens.size(1)
        text_padding = torch.arange(max_len)[None, :] >= lengths[:, None]           # [B, T]
        image_padding = torch.zeros(lengths.size(0), 1, dtype=torch.bool)           # [B, 1]
        key_padding_mask = torch.cat([image_padding, text_padding], dim=1).to(tokens.device)

        # Scaled dot-product self-attention, residual connection and LayerNorm.
        context = self.attention(self.q_proj(tokens), self.k_proj(tokens),
                                 self.v_proj(tokens), key_padding_mask)
        fused = self.norm(tokens + self.out_proj(context))

        # Mean-pool over real tokens only, then classify.
        valid = (~key_padding_mask).unsqueeze(-1).to(fused.dtype)       # [B, 1 + T, 1]
        pooled = (fused * valid).sum(dim=1) / valid.sum(dim=1)
        return self.classifier(pooled)                                  # [B, num_classes]


# =============================================================================
# PART 3: Instrumented Training Loop
# =============================================================================
def train_pipeline_step(model, optimizer, criterion, batch_data, rank, step=None):
    """One optimisation step. Every rank trains; only rank 0 writes telemetry to MLflow."""
    # Unpack data and move it to the model's device. batch_data[2] (lengths) is already
    # encoded inside the PackedSequence, so it is not needed here.
    device = next(model.parameters()).device
    images, packed_sequences = batch_data[0], batch_data[1]
    images = images.to(device, non_blocking=True)
    packed_sequences = packed_sequences.to(device)

    # zero_grad -> forward -> loss -> backward -> step
    optimizer.zero_grad()
    outputs = model(images, packed_sequences)
    # Dummy target for loss calculation
    target = torch.ones(outputs.shape[0], dtype=torch.long, device=device)
    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()

    # Detach before telemetry: .item() returns a plain Python float, so nothing logged or
    # returned keeps the autograd graph (and its activations) alive.
    loss_value = loss.item()

    # Optional global mean across ranks. Collective ops must run on EVERY rank, so this sits
    # OUTSIDE the rank shield (putting it inside would deadlock the other ranks).
    if dist.is_available() and dist.is_initialized():
        loss_tensor = torch.tensor(loss_value, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_value = loss_tensor.item() / dist.get_world_size()

    # Rank shield: only the master process (rank 0) writes to the tracking server.
    if rank == 0:
        mlflow.log_metric("train_loss", loss_value, step=step)

    return loss_value


# =============================================================================
# Runnable entry point (synthetic data stands in for the enterprise streams)
# =============================================================================
class SyntheticMultimodalDataset(Dataset):
    """Random RGB product frames paired with variable-length descriptive-log feature sequences."""

    def __init__(self, num_samples=32, image_size=224, seq_input_dim=32,
                 min_len=3, max_len=20, seed=0):
        generator = torch.Generator().manual_seed(seed)
        self.samples = []
        for _ in range(num_samples):
            raw = torch.randint(0, 256, (3, image_size, image_size),
                                dtype=torch.uint8, generator=generator)
            length = int(torch.randint(min_len, max_len + 1, (1,), generator=generator))
            sequence = torch.randn(length, seq_input_dim, generator=generator)
            self.samples.append((TF.to_pil_image(raw), sequence))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pil_image, sequence = self.samples[idx]
        return image_transform(pil_image), sequence


def main(num_epochs=1, batch_size=4, hidden_dim=128, n_heads=4, seq_input_dim=32):
    rank = get_rank()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    use_cuda = torch.cuda.is_available()

    if distributed:
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
        if use_cuda:
            torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    torch.manual_seed(42)

    dataset = SyntheticMultimodalDataset(seq_input_dim=seq_input_dim)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None),
                        sampler=sampler, collate_fn=custom_collate_fn)

    model = UnifiedMultiDomainModel(hidden_dim=hidden_dim, n_heads=n_heads,
                                    seq_input_dim=seq_input_dim).to(device)
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if use_cuda else None)

    # Only the unfrozen head (projections, LSTM, attention, classifier) is optimised.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    if rank == 0:
        mlflow.start_run(run_name="unified-multidomain-baseline")
        mlflow.log_params({"hidden_dim": hidden_dim, "n_heads": n_heads,
                           "d_k": hidden_dim // n_heads, "batch_size": batch_size,
                           "world_size": world_size,
                           "trainable_params": sum(p.numel() for p in trainable)})
    try:
        model.train()
        global_step = 0
        for epoch in range(num_epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch in loader:
                loss_value = train_pipeline_step(model, optimizer, criterion, batch,
                                                 rank, step=global_step)
                if rank == 0:
                    print(f"epoch {epoch} step {global_step} loss {loss_value:.4f}")
                global_step += 1
    finally:
        if rank == 0:
            mlflow.end_run()
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
