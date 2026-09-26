"""按顺序阅读：配置 → RMSNorm → RoPE → Attention → SwiGLU → Block → Llama。

这是保留 Llama 3 核心结构的教学模型，使用随机初始化的小尺寸参数。
运行：.venv/bin/python mini_llama.py
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Config:
    vocab_size: int = 32
    dim: int = 64
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2
    hidden_dim: int = 192
    max_seq_len: int = 128
    norm_eps: float = 1e-6
    rope_theta: float = 500_000.0

    def __post_init__(self):
        assert self.dim % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
        assert (self.dim // self.n_heads) % 2 == 0


# 1. 对每个 token 的最后一维进行归一化。
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_float = x.float()
        mean_square = x_float.square().mean(dim=-1, keepdim=True)
        normalized = x_float * torch.rsqrt(mean_square + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


# 2. 对 Q、K 的相邻两个特征做旋转，注入位置信息。
class RoPE(nn.Module):
    def __init__(self, head_dim, theta):
        super().__init__()
        inv_freq = theta ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q, k, start_pos=0):
        # q: [B, T, H, d]，k: [B, T, H_kv, d]
        length = q.shape[1]
        positions = torch.arange(
            start_pos, start_pos + length, device=q.device, dtype=torch.float32
        )
        angles = positions[:, None] * self.inv_freq[None, :]
        cos = angles.cos()[None, :, None, :]  # [1, T, 1, d/2]
        sin = angles.sin()[None, :, None, :]

        def rotate(x):
            even = x.float()[..., 0::2]
            odd = x.float()[..., 1::2]
            rotated = torch.stack(
                (even * cos - odd * sin, even * sin + odd * cos), dim=-1
            )
            return rotated.flatten(-2).to(x.dtype)

        return rotate(q), rotate(k)


# 3. GQA + RoPE + 因果注意力；显式写出矩阵运算便于理解。
class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.rope = RoPE(self.head_dim, cfg.rope_theta)

    def forward(self, x):
        batch, length, _ = x.shape
        q = self.wq(x).view(batch, length, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, length, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(batch, length, self.n_kv_heads, self.head_dim)

        q, k = self.rope(q, k)
        q = q.transpose(1, 2)  # [B, H, T, d]
        k = k.transpose(1, 2)  # [B, H_kv, T, d]
        v = v.transpose(1, 2)

        # 每组查询头共享同一个 K/V 头；显式重复便于学习。
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        scores = (q @ k.transpose(-2, -1)) / self.head_dim**0.5
        future_mask = torch.ones(
            length, length, dtype=torch.bool, device=x.device
        ).triu(diagonal=1)
        scores = scores.masked_fill(future_mask, float("-inf"))
        probabilities = F.softmax(scores.float(), dim=-1).to(q.dtype)
        output = probabilities @ v  # [B, H, T, d]
        output = output.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.wo(output)  # [B, T, D]


# 4. 每个 token 独立经过带门控的前馈网络。
class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.down = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# 5. Pre-Norm：先归一化，再运算，再加回残差。
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attention = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.feed_forward = SwiGLU(cfg)

    def forward(self, x):
        x = x + self.attention(self.attn_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


# 6. 从 token 编号到词表分数。模型输出 logits，不在这里做 softmax。
class MiniLlama(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def forward(self, token_ids):
        if token_ids.shape[1] > self.cfg.max_seq_len:
            raise ValueError("输入长度超过 max_seq_len")
        x = self.embedding(token_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens, temperature=0.0):
        if token_ids.ndim != 2 or token_ids.shape[1] == 0:
            raise ValueError("token_ids 应为非空的 [B, T] 张量")
        if max_new_tokens < 0 or temperature < 0:
            raise ValueError("max_new_tokens 和 temperature 必须非负")
        if token_ids.shape[1] + max_new_tokens > self.cfg.max_seq_len:
            raise ValueError("生成后的总长度超过 max_seq_len")

        # 基础版本每轮重新计算完整前缀，便于先理解自回归生成。
        for _ in range(max_new_tokens):
            logits = self(token_ids)[:, -1, :]
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probabilities = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids


# 7. 用简单循环序列验证模型能学习“预测下一个 token”。
def demo():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    cfg = Config()
    model = MiniLlama(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # 八条序列，具有不同的起点，但都遵循 0→1→…→7→0 的规律。
    sequence = (torch.arange(17)[None, :] + torch.arange(8)[:, None]) % 8
    inputs = sequence[:, :-1]   # [8, 16]
    targets = sequence[:, 1:]  # [8, 16]，相对输入左移一位

    print("parameters:", sum(p.numel() for p in model.parameters()))
    print("input shape:", tuple(inputs.shape))
    model.train()
    for step in range(101):
        logits = model(inputs)  # [8, 16, 32]
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), targets.reshape(-1)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in (0, 25, 50, 100):
            print(f"step {step:3d} | loss {loss.item():.4f}")

    model.eval()
    prompt = torch.tensor([[0, 1, 2]])
    print("logits shape:", tuple(model(inputs).shape))
    print("generated:", model.generate(prompt, max_new_tokens=13).tolist())


if __name__ == "__main__":
    demo()
