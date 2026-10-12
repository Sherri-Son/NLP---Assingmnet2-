import math
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    ):
        if lr < 0:
            raise ValueError("Learning rate must be nonnegative")
        if eps < 0:
            raise ValueError("eps must be nonnegative")
        if weight_decay < 0:
            raise ValueError("Weight decay must be nonnegative")
        if not all(0 <= beta < 1 for beta in betas):
            raise ValueError("Each beta must be in [0, 1)")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["t"] += 1
                t = state["t"]
                m = state["m"]
                v = state["v"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                adjusted_lr = (
                    lr * math.sqrt(1 - beta2 ** t)
                    / (1 - beta1 ** t)
                )

                p.addcdiv_(m, v.sqrt() + eps, value=-adjusted_lr)
                p.mul_(1 - lr * weight_decay)

        return loss

def get_lr_cosine_schedule(
    it,
    max_learning_rate,
    min_learning_rate,
    warmup_iters,
    cosine_cycle_iters,
):
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate

    if it <= cosine_cycle_iters:
        progress = (
            (it - warmup_iters)
            / (cosine_cycle_iters - warmup_iters)
        )

        return min_learning_rate + 0.5 * (
            1 + math.cos(math.pi * progress)
        ) * (max_learning_rate - min_learning_rate)

    return min_learning_rate

@torch.no_grad()
def gradient_clipping(parameters, max_l2_norm):
    grads = [
        p.grad for p in parameters
        if p.grad is not None
    ]

    if not grads:
        return

    total_squared_norm = sum(
        (grad ** 2).sum() for grad in grads
    )
    total_norm = torch.sqrt(total_squared_norm)

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)

        for grad in grads:
            grad.mul_(scale)