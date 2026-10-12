import csv
from pathlib import Path

import numpy as np
import torch

from data_utils import load_memmap_dataset
from student.data import get_batch
from student.model import TransformerLM, cross_entropy
from student.optimizer import (
    AdamW,
    get_lr_cosine_schedule,
    gradient_clipping,
)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training device:", device, flush=True)

    dataset = load_memmap_dataset(
        "data/tinystories_valid.bin",
        "data/tinystories_valid.meta",
    )

    config = {
        "vocab_size": 10000,
        "context_length": 64,
        "d_model": 128,
        "num_layers": 2,
        "num_heads": 4,
        "d_ff": 384,
        "rope_theta": 10000.0,
    }

    assert 0 <= dataset.min()
    assert dataset.max() < config["vocab_size"]

    model = TransformerLM(**config, device=device)
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )

    batch_size = 16
    num_steps = 500
    history = []

    for step in range(num_steps):
        lr = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=1e-3,
            min_learning_rate=1e-4,
            warmup_iters=20,
            cosine_cycle_iters=num_steps - 1,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(
            dataset,
            batch_size,
            config["context_length"],
            device,
        )

        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits, y)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")

        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm=1.0)
        optimizer.step()

        loss_value = loss.item()
        history.append((step + 1, loss_value))

        if step == 0 or (step + 1) % 25 == 0:
            print(
                f"Batch {step + 1}/{num_steps} | "
                f"loss={loss_value:.4f} | lr={lr:.6f}",
                flush=True,
            )

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    with (output_dir / "training_loss.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batches_processed", "loss"])
        writer.writerows(history)

    torch.save(
        {"config": config, "model_state_dict": model.state_dict()},
        output_dir / "model.pt",
    )

    print("Saved loss history and trained model.", flush=True)


if __name__ == "__main__":
    main()