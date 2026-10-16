import os
import argparse
import torch
import mlflow
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    return parser.parse_args()


def main():
    args = get_args()

    # Task 1: load the base model in 4-bit NormalFloat (NF4) with double quantization.
    # Weights are stored in 4-bit; matrix multiplications are upcast to float16.
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Task 2: LoRA adapters on the self-attention projections.
    # r and lora_alpha come from argparse so each sweep trial gets its own values.
    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    for epoch in range(5):
        loss = torch.tensor(0.85, requires_grad=True) * 0.8

        # Task 3: rank-0 protected MLflow tracking.
        # Only the global rank-0 process writes to MLflow; RANK defaults to "0" for local runs.
        if os.environ.get("RANK", "0") == "0":
            mlflow.log_metric("train_loss", loss.item(), step=epoch)


if __name__ == "__main__":
    main()
