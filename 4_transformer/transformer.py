"""Transformer 模型。

B: batch 大小，S/T: 源/目标长度，D: 隐藏维度，H: 头数。
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TransformerConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 64
    n_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    max_len: int = 128
    pad_id: int = 0

    def __post_init__(self) -> None:
        for name in (
            "src_vocab_size", "tgt_vocab_size", "d_model", "n_heads",
            "num_encoder_layers", "num_decoder_layers", "d_ff", "max_len",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须大于 0")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须在 [0, 1) 内")
        if not 0 <= self.pad_id < min(self.src_vocab_size, self.tgt_vocab_size):
            raise ValueError("pad_id 必须同时属于源词表和目标词表")


def make_padding_mask(tokens: Tensor, pad_id: int) -> Tensor:
    """屏蔽 PAD，True 表示可以关注。形状：[B, L] → [B, 1, 1, L]。"""
    if tokens.ndim != 2 or tokens.size(1) == 0:
        raise ValueError("tokens 必须是形状为 [batch, 非零长度] 的张量")
    # 每个头、每个 query 共用这份 PAD mask
    return (tokens != pad_id)[:, None, None, :]


def make_causal_mask(length: int, device: torch.device | None = None) -> Tensor:
    """只允许看当前位置和前面的位置，形状为 [1, 1, T, T]。"""
    return torch.ones(length, length, dtype=torch.bool, device=device).tril()[None, None]


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[Tensor, Tensor]:
    """计算 softmax(QKᵀ / √d_k)V，返回结果和 dropout 前的权重。

    Q: [B, H, Lq, d_k]，K/V: [B, H, Lk, d_k]。
    mask 用 bool，可广播到 [B, H, Lq, Lk]，True 表示保留。
    """
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("注意力 mask 必须是 bool，True 表示允许关注")
        scores = scores.masked_fill(~mask, float("-inf"))
        # 整行都是 -inf 时，softmax 会得到 NaN，所以先置零
        # softmax 后再把这些位置的权重清零
        scores = scores.masked_fill(~mask.any(dim=-1, keepdim=True), 0.0)
    weights = torch.softmax(scores, dim=-1)
    if mask is not None:
        weights = weights.masked_fill(~mask, 0.0)
    used_weights = dropout(weights) if dropout is not None else weights
    return used_weights @ value, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError("d_model 和 n_heads 必须为正，且 d_model 能被 n_heads 整除")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        # [B, L, D] → [B, L, H, d_k] → [B, H, L, d_k]
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None,
    ) -> Tensor:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))
        context, _ = scaled_dot_product_attention(q, k, v, mask, self.attention_dropout)
        # [B, H, Lq, d_k] → [B, Lq, H, d_k] → [B, Lq, D]
        batch, _, query_length, _ = context.shape
        context = context.transpose(1, 2).contiguous().view(batch, query_length, -1)
        return self.out_proj(context)


class LayerNorm(nn.Module):
    """每个 token 按最后一维做归一化。"""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        # 方差除以 D，不是 D - 1
        variance = (x - mean).square().mean(dim=-1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight * normalized + self.bias


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * frequency)
        # d_model 为奇数时，cos 比 sin 少一个位置
        pe[:, 1::2] = torch.cos(position * frequency[: d_model // 2])
        # 位置编码不用训练，但要跟着模型保存和移动设备
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, D]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"序列长度 {x.size(1)} 超过 max_len={self.pe.size(1)}")
        return self.dropout(x + self.pe[:, : x.size(1)].to(dtype=x.dtype))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # [B, L, D] → [B, L, d_ff] → [B, L, D]，每个位置用同一组参数
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(config.d_model, config.n_heads, config.dropout)
        self.feed_forward = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout)
        self.norm1 = LayerNorm(config.d_model)
        self.norm2 = LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, src_mask: Tensor) -> Tensor:
        # 先做残差相加，再做 LayerNorm（Post-LN）
        attended = self.self_attention(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attended))
        return self.norm2(x + self.dropout(self.feed_forward(x)))


class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(config.d_model, config.n_heads, config.dropout)
        self.cross_attention = MultiHeadAttention(config.d_model, config.n_heads, config.dropout)
        self.feed_forward = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout)
        self.norm1 = LayerNorm(config.d_model)
        self.norm2 = LayerNorm(config.d_model)
        self.norm3 = LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, memory: Tensor, tgt_mask: Tensor, src_mask: Tensor) -> Tensor:
        attended = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attended))
        # Q 用 Decoder 的输出，K/V 用 Encoder 的输出，两边长度可以不同
        attended = self.cross_attention(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(attended))
        return self.norm3(x + self.dropout(self.feed_forward(x)))


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.src_embedding = nn.Embedding(config.src_vocab_size, config.d_model, config.pad_id)
        self.tgt_embedding = nn.Embedding(config.tgt_vocab_size, config.d_model, config.pad_id)
        self.position = SinusoidalPositionalEncoding(config.d_model, config.max_len, config.dropout)
        self.encoder_layers = nn.ModuleList(
            EncoderLayer(config) for _ in range(config.num_encoder_layers)
        )
        self.decoder_layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(config.num_decoder_layers)
        )
        self.output_projection = nn.Linear(config.d_model, config.tgt_vocab_size)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        # 初始化后把 PAD 对应的向量清零
        with torch.no_grad():
            self.src_embedding.weight[self.config.pad_id].zero_()
            self.tgt_embedding.weight[self.config.pad_id].zero_()

    def encode(self, src: Tensor) -> tuple[Tensor, Tensor]:
        src_mask = make_padding_mask(src, self.config.pad_id)
        x = self.position(self.src_embedding(src) * math.sqrt(self.config.d_model))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x, src_mask  # memory: [B, S, D]

    def decode(self, tgt: Tensor, memory: Tensor, src_mask: Tensor) -> Tensor:
        tgt_mask = make_padding_mask(tgt, self.config.pad_id)
        tgt_mask = tgt_mask & make_causal_mask(tgt.size(1), tgt.device)
        x = self.position(self.tgt_embedding(tgt) * math.sqrt(self.config.d_model))
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, src_mask)
        # 这里不用 softmax，交叉熵会处理
        return self.output_projection(x)  # [B, T, tgt_vocab_size]

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        memory, src_mask = self.encode(src)
        return self.decode(tgt, memory, src_mask)


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: Tensor,
    max_new_tokens: int,
    bos_id: int = 1,
    eos_id: int = 2,
) -> Tensor:
    """从 BOS 开始，每次选概率最大的 token。

    返回结果包含 BOS，遇到 EOS 的序列后面补 PAD。
    每步都重新计算 Decoder，没有做 KV cache。
    """
    if not 1 <= max_new_tokens <= model.config.max_len:
        raise ValueError("max_new_tokens 必须在 [1, model.config.max_len] 内")
    if not 0 <= bos_id < model.config.tgt_vocab_size or not 0 <= eos_id < model.config.tgt_vocab_size:
        raise ValueError("bos_id 和 eos_id 必须属于目标词表")
    if len({bos_id, eos_id, model.config.pad_id}) != 3:
        raise ValueError("BOS、EOS 和 PAD 必须使用不同的 token ID")
    was_training = model.training
    model.eval()
    try:
        memory, src_mask = model.encode(src)
        generated = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_new_tokens):
            logits = model.decode(generated, memory, src_mask)[:, -1, :]
            # 生成时不选 BOS 和 PAD
            logits[:, [bos_id, model.config.pad_id]] = float("-inf")
            next_token = logits.argmax(dim=-1)
            next_token = next_token.masked_fill(finished, model.config.pad_id)
            generated = torch.cat((generated, next_token[:, None]), dim=1)
            finished = finished | (next_token == eos_id)
            if finished.all():
                break
        return generated
    finally:
        model.train(was_training)
