"""Mini LLaMA 模型。B/T/D/H 分别是 batch 大小、序列长度、维度和头数。"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 2
    hidden_dim: int | None = None
    max_seq_len: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 500_000.0

    def __post_init__(self):
        for name in ("vocab_size", "dim", "n_layers", "n_heads", "n_kv_heads", "max_seq_len"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if self.dim % self.n_heads:
            raise ValueError("dim 必须能被 n_heads 整除")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads 必须能被 n_kv_heads 整除")
        if (self.dim // self.n_heads) % 2:
            raise ValueError("RoPE 要求每个头的维度是偶数")
        if self.hidden_dim is not None and (type(self.hidden_dim) is not int or self.hidden_dim <= 0):
            raise ValueError("hidden_dim 必须是正整数")
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("norm_eps 必须为有限正数")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("rope_theta 必须为有限正数")

    @property
    def ffn_dim(self) -> int:
        # SwiGLU 有三组权重，用 8D/3 控制参数量，再对齐到 32
        return self.hidden_dim if self.hidden_dim is not None else math.ceil(8 * self.dim / 3 / 32) * 32


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # 不减均值，用 float32 算均方值，减少低精度计算的溢出问题
        normalized = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


def rope_angles(positions: Tensor, head_dim: int, theta: float) -> tuple[Tensor, Tensor]:
    """每两个维度共用一个频率，cos/sin 的形状是 [1, 1, T, Dh/2]。"""
    frequencies = theta ** (-torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / head_dim)
    angles = positions.float()[:, None] * frequencies[None, :]
    return angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # 两个相邻维度做一次旋转：[x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
    even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2).to(x.dtype)


# 缓存保留原来的 KV 头数：[B, Hkv, 缓存长度, Dh]
# 缓存里的 K 已经旋转过，后面只旋转新 K
KVCache = tuple[Tensor, Tensor]
ModelCache = tuple[KVCache, ...]


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.q_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.k_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, blocked: Tensor,
                past: KVCache | None = None, use_cache: bool = False) -> tuple[Tensor, KVCache | None]:
        batch, length, dim = x.shape
        # 把 Q/K/V 拆成多个头，形状变成 [B, 头数, T, Dh]
        q = self.q_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if past is not None:
            if past[0].dtype != k.dtype or past[1].dtype != v.dtype:
                raise ValueError("缓存 dtype 必须与当前 K/V 相同")
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        present = (k, v) if use_cache else None

        # 4 个 Q 头、2 个 KV 头时，每两个相邻的 Q 头共用一组 K/V
        repeats = self.n_heads // self.n_kv_heads
        keys = k.repeat_interleave(repeats, dim=1)
        values = v.repeat_interleave(repeats, dim=1)
        # [B,H,T,Dh] @ [B,H,Dh,S] -> [B,H,T,S]，S 是缓存长度加当前长度
        scores = (q @ keys.transpose(-1, -2)).float() / math.sqrt(self.head_dim)
        scores = scores.masked_fill(blocked, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1).to(q.dtype)
        attended = probabilities @ values
        merged = attended.transpose(1, 2).contiguous().view(batch, length, dim)
        return self.o_proj(merged), present


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.gate_proj(x)
        # SiLU(g) = g * sigmoid(g)，再乘另一条分支的结果
        return self.down_proj((gate * torch.sigmoid(gate)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.feed_forward = SwiGLU(config)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, blocked: Tensor,
                past: KVCache | None, use_cache: bool) -> tuple[Tensor, KVCache | None]:
        attention, present = self.attention(self.attention_norm(x), cos, sin, blocked, past, use_cache)
        x = x + attention
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, present


@dataclass
class ModelOutput:
    logits: Tensor
    loss: Tensor | None = None
    past_key_values: ModelCache | None = None


class MiniLlama(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _past_length(self, past: ModelCache | None, tokens: Tensor) -> int:
        if past is None:
            return 0
        if len(past) != self.config.n_layers:
            raise ValueError("每层都必须有一对 K/V 缓存")
        lengths = set()
        for k, v in past:
            expected = (tokens.shape[0], self.config.n_kv_heads)
            if k.ndim != 4 or v.shape != k.shape or k.shape[:2] != expected:
                raise ValueError("缓存形状应为 [batch, n_kv_heads, past_length, head_dim]")
            if k.shape[-1] != self.config.dim // self.config.n_heads:
                raise ValueError("缓存 head_dim 不匹配")
            if k.device != tokens.device or v.device != tokens.device:
                raise ValueError("缓存与输入必须在同一个设备")
            lengths.add(k.shape[2])
        if len(lengths) != 1:
            raise ValueError("各层缓存的长度必须一致")
        return lengths.pop()

    def forward(self, tokens: Tensor, targets: Tensor | None = None, *,
                past_key_values: ModelCache | None = None, use_cache: bool = False) -> ModelOutput:
        """每个位置预测下一个字符，targets 要提前错开一位。"""
        if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] == 0:
            raise ValueError("tokens 必须是非空的 [batch, length] 张量")
        if tokens.dtype != torch.long:
            raise ValueError("tokens 必须使用 torch.long")
        if past_key_values is not None and not use_cache:
            raise ValueError("传入缓存时必须设置 use_cache=True")
        if use_cache and (torch.is_grad_enabled() or targets is not None):
            raise ValueError("KV cache 仅用于推理，请使用 torch.no_grad() 或 torch.inference_mode()，且不传 targets")
        if targets is not None and (targets.shape != tokens.shape or targets.dtype != torch.long):
            raise ValueError("targets 必须与 tokens 形状相同，且使用 torch.long")
        start = self._past_length(past_key_values, tokens)
        length = tokens.shape[1]
        if start + length > self.config.max_seq_len:
            raise ValueError(f"输入与缓存总长度超过 max_seq_len={self.config.max_seq_len}")

        positions = torch.arange(start, start + length, device=tokens.device)
        cos, sin = rope_angles(positions, self.config.dim // self.config.n_heads, self.config.rope_theta)
        key_positions = torch.arange(start + length, device=tokens.device)
        # 按绝对位置屏蔽未来 token，True 表示不能看
        # 算上缓存长度，才能正确处理一次输入多个新 token 的情况
        blocked = key_positions[None, :] > positions[:, None]
        x = self.embedding(tokens)
        new_cache = []
        for index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[index]
            x, present = layer(x, cos, sin, blocked, past, use_cache)
            if present is not None:
                new_cache.append(present)
        logits = self.lm_head(self.norm(x)).float()  # [B,T,V]，这里不做 softmax
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return ModelOutput(logits, loss, tuple(new_cache) if use_cache else None)
