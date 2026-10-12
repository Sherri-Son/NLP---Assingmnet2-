import torch
from torch import nn
import math

class Linear(nn.Module):
  def __init__(self, in_features, out_features, device = None, dtype = None):
    super().__init__()

    self.weight = nn.Parameter(
        torch.empty(out_features, in_features, device = device, dtype = dtype)
    )
    std = math.sqrt(2.0 / (in_features + out_features))
    nn.init.trunc_normal_(
        self.weight,
        mean = 0.0,
        std = std,
        a = -3 * std,
        b = 3 * std
    )

  def forward(self, x):
    return x @ self.weight.t()

class Embedding(nn.Module):
  def __init__(self, num_embeddings, embedding_dim, device = None, dtype = None):
    super().__init__()

    self.weight = nn.Parameter(
        torch.empty(
            num_embeddings, embedding_dim,
            device = device, dtype = dtype)
    )

    nn.init.trunc_normal_(

        self.weight,
        mean = 0.0,
        std = 1.0,
        a = -3,
        b = 3
    )

  def forward(self, token_ids):
      return self.weight[token_ids]

class RMSNorm(nn.Module):
  def __init__(self, d_model, eps = 1e-5, device = None, dtype = None):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(
        torch.ones(d_model, device = device, dtype = dtype
    ))
  def forward(self, x):
      in_dtype = x.dtype
      x = x.to(torch.float32)

      mean_square = (x * x).mean(dim=-1, keepdim=True)
      rms = torch.sqrt(mean_square + self.eps)
      result = (x / rms) * self.weight

      return result.to(in_dtype)


def silu(x):
    return x * torch.sigmoid(x)


def softmax(x, dim):
    max_values = x.max(dim=dim, keepdim=True).values
    shifted = x - max_values

    exp_values = torch.exp(shifted)
    return exp_values / exp_values.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]

    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attention_weights = softmax(scores, dim=-1)

    return attention_weights @ V

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for pairwise rotations")
        pair_indices = torch.arange(
            0, d_k, 2, device=device, dtype=torch.float32
        )
        inv_freq = theta ** (-pair_indices / d_k)
        positions = torch.arange(
            max_seq_len, device=device, dtype=torch.float32
        )
        angles = positions[:, None] * inv_freq[None, :]
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x, token_positions):
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack(
            (rotated_even, rotated_odd), dim=-1
        ).flatten(-2).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = 64 * math.ceil((8 * d_model) / (3 * 64))
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        gate = silu(self.w1(x))
        values = self.w3(x)
        return self.w2(gate * values)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, d_model, num_heads,
        max_seq_len=None, theta=None,
        device=None, dtype=None
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(
            d_model, d_model, device=device, dtype=dtype
        )

        self.rope = None
        if theta is not None:
            if max_seq_len is None:
                raise ValueError("max_seq_len is required for RoPE")

            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=device,
            )

    def forward(self, x, token_positions=None):
        seq_len = x.shape[-2]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        head_shape = (*x.shape[:-1], self.num_heads, self.d_k)

        q = q.reshape(head_shape).transpose(-3, -2)
        k = k.reshape(head_shape).transpose(-3, -2)
        v = v.reshape(head_shape).transpose(-3, -2)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(
                    seq_len, device=x.device
                )

            positions = token_positions.unsqueeze(-2)
            q = self.rope(q, positions)
            k = self.rope(k, positions)

        indices = torch.arange(seq_len, device=x.device)
        causal_mask = indices[:, None] >= indices[None, :]

        attended = scaled_dot_product_attention(
            q, k, v, mask=causal_mask
        )

        merged = attended.transpose(-3, -2).reshape(x.shape)

        return self.output_proj(merged)

class TransformerBlock(nn.Module):
    def __init__(
        self, d_model, num_heads, d_ff,
        max_seq_len, theta,
        device=None, dtype=None
    ):
        super().__init__()

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)

        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )

        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

        self.ffn = SwiGLU(
            d_model, d_ff, device=device, dtype=dtype
        )

    def forward(self, x, token_positions=None):
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(
        self, vocab_size, context_length, d_model,
        num_layers, num_heads, d_ff, rope_theta,
        device=None, dtype=None
    ):
        super().__init__()

        self.context_length = context_length

        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(
            d_model, device=device, dtype=dtype
        )

        self.lm_head = Linear(
            d_model, vocab_size, device=device, dtype=dtype
        )

    def forward(self, token_ids):
        if token_ids.shape[-1] > self.context_length:
            raise ValueError("Input exceeds the model's context length")

        x = self.token_embeddings(token_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        return self.lm_head(x)

def cross_entropy(inputs, targets):
    shifted = inputs - inputs.max(dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))
    target_logits = shifted.gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)
    return (log_sum_exp - target_logits).mean()
