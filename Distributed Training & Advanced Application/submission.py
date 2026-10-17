"""
submission.py - Distributed Training Scenario (DeepSpeed ZeRO-3 + Azure ML SDK v2)

Scales a ~1.5-billion-parameter transformer from a single device to a 2-node x 4-GPU
Azure ML cluster. The three required components are combined into this one file:

    PART 0  Concept verification   runs on CPU in Jupyter, no GPU needed -> prints SUCCESS
    PART 1  train_deepspeed.py     ZeRO-3 training script with activation checkpointing
    PART 2  ds_config.json         DeepSpeed configuration, embedded as the dict `ds_config`
    PART 3  submit_job.py          Azure ML SDK v2 multi-node command job

Usage
    python submission.py                  # default: run the concept verification (CPU only)
    python submission.py --mode export    # write ds_config.json + train_deepspeed.py launcher
    python submission.py --mode submit    # export, then submit the 2-node job to Azure ML
    python submission.py --mode train     # distributed training entry point (runs on the cluster)

    In Jupyter:  %run submission.py

Requirements: torch (all modes); deepspeed (train); azure-ai-ml + azure-identity (submit).
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset

try:
    import deepspeed
except ImportError:  # verification mode only needs torch
    deepspeed = None

try:
    from azure.ai.ml import MLClient, command, PyTorchDistribution
    from azure.identity import DefaultAzureCredential
except ImportError:  # the training container does not need the Azure ML SDK
    MLClient = command = PyTorchDistribution = DefaultAzureCredential = None


# Cluster topology shared by the config (Part 2) and the job definition (Part 3).
NUM_NODES = 2
GPUS_PER_NODE = 4
WORLD_SIZE = NUM_NODES * GPUS_PER_NODE  # 8 ranks
DS_CONFIG_PATH = "ds_config.json"


# =====================================================================================
# PART 0: Concept verification (provided script; classes prefixed "Verify" so they do
#         not collide with the production LargeModel in Part 1). Runs on CPU.
# =====================================================================================
class VerifyTransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(256, 256)
        self.ff = nn.Linear(256, 1024)
        self.ff_out = nn.Linear(1024, 256)

    def forward(self, x):
        return self.ff_out(self.ff(self.attn(x)))


class VerifyLargeModel(nn.Module):
    def __init__(self, num_layers=8):
        super().__init__()
        self.layers = nn.ModuleList([VerifyTransformerLayer() for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)
        return x


def run_activation_checkpointing_demo():
    torch.manual_seed(0)
    model = VerifyLargeModel(num_layers=8)
    x = torch.randn(2, 32, 256, requires_grad=True)
    output = model(x)
    loss = output.mean()
    loss.backward()
    print(f"Forward pass output shape: {tuple(output.shape)}")
    print(f"Loss value: {loss.item():.6f}")
    print("Backward pass completed via recomputation.")
    return output, loss


def run_zero3_sharding_math(total_params=1_500_000_000, num_gpus=8):
    bytes_per_param_full_state = 16
    total_state_gb = (total_params * bytes_per_param_full_state) / (1024 ** 3)
    sharded_gb_per_gpu = total_state_gb / num_gpus
    print(f"Model parameters: {total_params:,}")
    print(f"Full training state without sharding: {total_state_gb:.2f} GB")
    print(f"Per-GPU state WITH ZeRO-3 sharding across {num_gpus} GPUs: {sharded_gb_per_gpu:.2f} GB")
    return total_state_gb, sharded_gb_per_gpu


def _checkpointed_grads_match_plain_grads():
    """Recomputation must be exact: checkpointed and non-checkpointed grads are identical."""
    torch.manual_seed(0)
    model = VerifyLargeModel(num_layers=8)
    x = torch.randn(2, 32, 256, requires_grad=True)

    model(x).mean().backward()  # checkpointed path
    ckpt_grads = [p.grad.detach().clone() for p in model.parameters()]

    model.zero_grad()
    out = x
    for layer in model.layers:  # plain path: every activation is stored
        out = layer(out)
    out.mean().backward()
    plain_grads = [p.grad for p in model.parameters()]

    return all(torch.allclose(a, b, atol=1e-6) for a, b in zip(ckpt_grads, plain_grads))


def run_verification():
    print("=== Activation checkpointing ===")
    output, loss = run_activation_checkpointing_demo()
    assert tuple(output.shape) == (2, 32, 256) and torch.isfinite(loss)
    assert _checkpointed_grads_match_plain_grads(), "checkpointed grads differ from plain grads"
    print("Checkpointed gradients match non-checkpointed gradients.")

    print("\n=== ZeRO-3 sharding math ===")
    full_gb, per_gpu_gb = run_zero3_sharding_math(num_gpus=WORLD_SIZE)
    assert abs(per_gpu_gb * WORLD_SIZE - full_gb) < 1e-9

    print("\n=== Batch-size arithmetic ===")
    effective = (ds_config["train_micro_batch_size_per_gpu"]
                 * ds_config["gradient_accumulation_steps"] * WORLD_SIZE)
    print(f"{ds_config['train_micro_batch_size_per_gpu']} micro x "
          f"{ds_config['gradient_accumulation_steps']} accum x {WORLD_SIZE} GPUs = {effective}")
    assert effective == ds_config["train_batch_size"], "train_batch_size inconsistent with cluster"
    assert ds_config["zero_optimization"]["stage"] == 3

    print("\nSUCCESS: Concepts verified.")


# =====================================================================================
# PART 1: train_deepspeed.py - DeepSpeed ZeRO-3 training with activation checkpointing
# =====================================================================================
@dataclass
class ModelConfig:
    vocab_size: int = 32000
    max_seq_len: int = 1024
    hidden_size: int = 2048
    num_heads: int = 16
    num_layers: int = 27      # with the sizes above: ~1.49B parameters
    ffn_mult: int = 4


class TransformerLayer(nn.Module):
    """Pre-norm decoder block: causal self-attention + GELU MLP, each with a residual."""

    def __init__(self, hidden_size, num_heads, ffn_mult=4):
        super().__init__()
        self.num_heads = num_heads
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.attn_out = nn.Linear(hidden_size, hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ff = nn.Linear(hidden_size, ffn_mult * hidden_size)
        self.ff_out = nn.Linear(ffn_mult * hidden_size, hidden_size)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = (self.qkv(self.ln_1(x))
                   .view(b, t, 3, self.num_heads, c // self.num_heads)
                   .permute(2, 0, 3, 1, 4))
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.attn_out(attn.transpose(1, 2).reshape(b, t, c))
        x = x + self.ff_out(F.gelu(self.ff(self.ln_2(x))))
        return x


class LargeModel(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.hidden_size)
        self.layers = nn.ModuleList([
            TransformerLayer(cfg.hidden_size, cfg.num_heads, cfg.ffn_mult)
            for _ in range(cfg.num_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids):
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        # Embedding output requires grad, so the reentrant checkpoint below propagates
        # gradients into every layer's parameters.
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        for layer in self.layers:
            # Activation checkpointing: keep only this layer's input; its internal
            # activations are discarded and recomputed during the backward pass.
            x = deepspeed.checkpointing.checkpoint(layer, x)
        return self.lm_head(self.ln_f(x))


class SyntheticTokenDataset(Dataset):
    """Deterministic random token sequences (next-token prediction) for the blueprint run."""

    def __init__(self, num_samples, seq_len, vocab_size, seed=1234):
        self.num_samples, self.seq_len, self.vocab_size, self.seed = (
            num_samples, seq_len, vocab_size, seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(self.seed + idx)
        tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,), generator=g)
        return tokens[:-1], tokens[1:]


def wrap_deepspeed_model(model, config_path=DS_CONFIG_PATH, training_data=None):
    """Wrap the model in DeepSpeed's engine. ZeRO-3 shards parameters, gradients and
    optimizer states across all ranks according to the config file."""
    model_engine, optimizer, train_loader, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config=config_path,
        training_data=training_data,  # DeepSpeed builds a DistributedSampler loader
    )
    return model_engine, optimizer, train_loader, lr_scheduler


def parse_train_args(argv=None):
    defaults = ModelConfig()
    p = argparse.ArgumentParser(description="DeepSpeed ZeRO-3 training")
    p.add_argument("--deepspeed_config", default=DS_CONFIG_PATH)
    p.add_argument("--max_steps", type=int, default=50, help="optimizer steps")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_dir", default=None,
                   help="optional ZeRO checkpoint dir, e.g. ./outputs/checkpoints")
    p.add_argument("--vocab_size", type=int, default=defaults.vocab_size)
    p.add_argument("--seq_len", type=int, default=defaults.max_seq_len)
    p.add_argument("--hidden_size", type=int, default=defaults.hidden_size)
    p.add_argument("--num_heads", type=int, default=defaults.num_heads)
    p.add_argument("--num_layers", type=int, default=defaults.num_layers)
    p.add_argument("--local_rank", type=int, default=-1)  # accepted if a launcher passes it
    args, _ = p.parse_known_args(argv)
    return args


def train_main(argv=None):
    if deepspeed is None:
        raise ImportError("DeepSpeed is required for training: pip install deepspeed")
    args = parse_train_args(argv)

    # 1) Distributed init. Azure ML's PyTorchDistribution sets MASTER_ADDR, MASTER_PORT,
    #    WORLD_SIZE, NODE_RANK, RANK and LOCAL_RANK; init_distributed() reads them.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(dist_backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # 2) Make sure the config file exists (it ships with the job; this is a fallback).
    if local_rank == 0 and not os.path.exists(args.deepspeed_config):
        write_ds_config(args.deepspeed_config)
    torch.distributed.barrier()
    with open(args.deepspeed_config) as f:
        cfg_dict = json.load(f)

    # 3) Build the model inside zero.Init so each rank only materialises its own shard
    #    (a full fp32 copy of 1.5B params would be ~6 GB per process before sharding).
    model_cfg = ModelConfig(vocab_size=args.vocab_size, max_seq_len=args.seq_len,
                            hidden_size=args.hidden_size, num_heads=args.num_heads,
                            num_layers=args.num_layers)
    with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config):
        model = LargeModel(model_cfg)
    n_params = sum(getattr(p, "ds_numel", p.numel()) for p in model.parameters())

    # 4) Wrap with DeepSpeed and get a distributed data loader.
    dataset = SyntheticTokenDataset(
        num_samples=cfg_dict["train_batch_size"] * args.max_steps,
        seq_len=args.seq_len, vocab_size=args.vocab_size)
    engine, _, train_loader, _ = wrap_deepspeed_model(model, args.deepspeed_config, dataset)

    if rank == 0:
        print(f"World size {world_size} | params {n_params / 1e6:,.1f}M | "
              f"ZeRO stage {engine.zero_optimization_stage()} | "
              f"global batch {engine.train_batch_size()}", flush=True)

    # 5) Training loop: engine.backward/step handle loss scaling, gradient
    #    reduce-scatter, optimizer step on the CPU-offloaded shard, and accumulation.
    engine.train()
    start = time.time()
    for micro_step, (input_ids, labels) in enumerate(train_loader, start=1):
        input_ids = input_ids.to(engine.device)
        labels = labels.to(engine.device)
        logits = engine(input_ids)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1))
        engine.backward(loss)
        engine.step()

        if rank == 0 and (micro_step == 1 or micro_step % args.log_every == 0):
            mem = (f" | peak GPU mem {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
                   if torch.cuda.is_available() else "")
            print(f"step {engine.global_steps:>5} | loss {loss.item():.4f} | "
                  f"{time.time() - start:.1f}s{mem}", flush=True)

    if args.save_dir:  # collective call: every rank writes its own shard
        engine.save_checkpoint(args.save_dir)
    if rank == 0:
        print(f"Training finished: {engine.global_steps} optimizer steps.", flush=True)


# =====================================================================================
# PART 2: ds_config.json - embedded as a Python dict (written to disk by write_ds_config)
# =====================================================================================
ds_config = {
    # Global batch: 4 micro-batch x 1 accumulation step x 8 GPUs (2 nodes x 4) = 32.
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 1,
    # Mixed precision: fp16 weights/grads for compute, fp32 master weights in the optimizer.
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        # Stage 3 shards parameters, gradients AND optimizer states across all ranks.
        "stage": 3,
        # Move optimizer states and parameters out of GPU VRAM into pinned CPU RAM;
        # pinned (page-locked) memory enables faster async host<->device copies.
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        # Additions to the template: overlap gradient reduction with backward compute,
        # and copy gradients into a contiguous buffer to avoid fragmentation.
        "overlap_comm": True,
        "contiguous_gradients": True
    },
    # Addition to the template: the engine needs an optimizer. With offload_optimizer
    # on CPU, DeepSpeed runs this as its vectorised DeepSpeedCPUAdam (AdamW mode).
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 1.5e-4, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1}
    },
    "gradient_clipping": 1.0,
    "steps_per_print": 10
}


def write_ds_config(path=DS_CONFIG_PATH, config=None):
    """Serialise the embedded dict to ds_config.json and confirm it is valid JSON."""
    config = ds_config if config is None else config
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    with open(path) as f:
        assert json.load(f) == config
    return path


TRAIN_LAUNCHER = '''"""train_deepspeed.py - generated by `python submission.py --mode export`.
Entry point for the Azure ML job; the training code is PART 1 of submission.py,
which is uploaded in the same code folder."""
import sys

from submission import train_main

if __name__ == "__main__":
    train_main(sys.argv[1:])
'''


def export_components(folder="."):
    """Write ds_config.json and the train_deepspeed.py launcher next to submission.py,
    so the code folder uploaded by the job contains everything the command needs."""
    cfg_path = write_ds_config(os.path.join(folder, DS_CONFIG_PATH))
    launcher_path = os.path.join(folder, "train_deepspeed.py")
    with open(launcher_path, "w") as f:
        f.write(TRAIN_LAUNCHER)
    print(f"Wrote {cfg_path} and {launcher_path}")


# =====================================================================================
# PART 3: submit_job.py - Azure ML SDK v2 multi-node command job
# =====================================================================================
def build_distributed_job():
    """Multi-node command job: 2 nodes x 4 GPUs = 8 DeepSpeed ranks."""
    distributed_job = command(
        code="./",                                    # folder with submission.py, launcher, config
        command="python train_deepspeed.py --deepspeed_config ds_config.json",
        # Curated Azure Container for PyTorch image: PyTorch, CUDA 12.1, DeepSpeed, NCCL.
        environment="azureml:acpt-pytorch-2.1-cuda12.1@latest",
        compute="gpu-cluster-V100",
        instance_count=2,                             # 2 nodes
        distribution=PyTorchDistribution(
            process_count_per_instance=4              # one process per GPU on each node
        ),
        environment_variables={
            "NCCL_DEBUG": "INFO",       # log the transport/topology NCCL selects
            "NCCL_IB_DISABLE": "0",     # keep InfiniBand (RDMA) on for inter-node traffic
            "NCCL_P2P_LEVEL": "NVL",    # direct GPU-to-GPU P2P only over NVLink
        },
        shm_size="16g",                 # room in /dev/shm for intra-node collectives
        experiment_name="deepspeed-zero3-1p5b",
        display_name="zero3-1p5b-2node-4gpu",
    )
    return distributed_job


def submit_distributed_job():
    if MLClient is None:
        raise ImportError("Install the Azure ML SDK v2: pip install azure-ai-ml azure-identity")
    export_components()  # ds_config.json + train_deepspeed.py must be in the code folder
    ml_client = MLClient.from_config(DefaultAzureCredential())
    distributed_job = build_distributed_job()
    returned_job = ml_client.jobs.create_or_update(distributed_job)
    print(f"Created job: {returned_job.name}")
    print(f"Monitor in Azure ML studio: {returned_job.studio_url}")
    return returned_job


# =====================================================================================
# Main execution block
# =====================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["verify", "export", "submit", "train"],
                        default="verify")
    cli_args, remaining = parser.parse_known_args()  # tolerant of Jupyter kernel args

    if cli_args.mode == "verify":
        run_verification()
    elif cli_args.mode == "export":
        export_components()
    elif cli_args.mode == "submit":
        submit_distributed_job()
    else:
        train_main(remaining)
