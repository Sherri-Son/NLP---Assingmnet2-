import numpy as np
import torch


def get_batch(dataset, batch_size, context_length, device):
    num_starts = len(dataset) - context_length

    if num_starts <= 0:
        raise ValueError("Dataset must contain more tokens than context_length")

    starts = np.random.randint(0, num_starts, size=batch_size)

    inputs = np.stack([
        dataset[start : start + context_length]
        for start in starts
    ])

    targets = np.stack([
        dataset[start + 1 : start + context_length + 1]
        for start in starts
    ])

    x = torch.tensor(inputs, dtype=torch.long, device=device)
    y = torch.tensor(targets, dtype=torch.long, device=device)

    return x, y