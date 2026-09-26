"""加载模型，接着输入的文字往下写。"""

import argparse
import math
from pathlib import Path

import torch
from torch import Tensor

from model import MiniLlama, ModelConfig
from tokenizer import CharTokenizer


def sample_next_token(logits: Tensor, temperature: float = 0.8, top_k: int | None = 20) -> Tensor:
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature 必须是有限的非负数；0 表示贪心解码")
    if top_k is not None and (type(top_k) is not int or top_k <= 0):
        raise ValueError("top_k 必须为正整数或 None")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    scores = logits.float() / temperature
    if top_k is not None:
        # 只从分数最高的 k 个 token 中采样，同分也不会多选
        values, indices = torch.topk(scores, min(top_k, scores.shape[-1]), dim=-1)
        choice = torch.multinomial(torch.softmax(values, dim=-1), num_samples=1)
        return indices.gather(-1, choice)
    return torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1)


@torch.inference_mode()
def generate_tokens(model: MiniLlama, prompt: Tensor, max_new_tokens: int = 80,
                    temperature: float = 0.8, top_k: int | None = 20,
                    use_cache: bool = True) -> Tensor:
    """返回原文和续写。同一批输入要等长，不处理 padding，也不靠 EOS 停止。"""
    if prompt.ndim != 2 or prompt.shape[0] == 0 or prompt.shape[1] == 0 or prompt.dtype != torch.long:
        raise ValueError("prompt 必须是非空的二维 torch.long 张量")
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens 必须是非负整数")
    if prompt.shape[1] + max_new_tokens > model.config.max_seq_len:
        raise ValueError(f"prompt 长度 + max_new_tokens 不能超过 {model.config.max_seq_len}")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature 必须是有限的非负数")
    if top_k is not None and (type(top_k) is not int or top_k <= 0):
        raise ValueError("top_k 必须为正整数或 None")

    was_training = model.training
    model.eval()
    try:
        generated = prompt.clone()
        current = prompt
        cache = None
        for _ in range(max_new_tokens):
            # 开启缓存时，第一次输入整个 prompt，之后每次只输入新 token
            output = model(current if use_cache else generated, past_key_values=cache, use_cache=use_cache)
            next_token = sample_next_token(output.logits[:, -1, :], temperature, top_k)
            generated = torch.cat((generated, next_token), dim=1)
            current, cache = next_token, output.past_key_values
        return generated
    finally:
        model.train(was_training)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[MiniLlama, CharTokenizer]:
    # 读取权重和配置，再用它们创建模型
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("不支持的 checkpoint 版本")
    tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])
    config = ModelConfig(**checkpoint["config"])
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError("checkpoint 的模型词表大小与 tokenizer 不一致")
    model = MiniLlama(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), tokenizer


def main():
    parser = argparse.ArgumentParser(description="Mini LLaMA 文本生成")
    parser.add_argument("--checkpoint", type=Path, default=Path("out/model.pt"))
    parser.add_argument("--prompt", default="春天来了，")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 = 贪心解码")
    parser.add_argument("--top-k", type=int, default=20, help="0 = 不限制候选数量")
    parser.add_argument("--no-cache", action="store_true", help="每一步重新计算完整前缀")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads 必须为正数")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("当前环境未检测到可用 CUDA，请使用 --device cpu")
    try:
        model, tokenizer = load_checkpoint(args.checkpoint, device)
        ids = tokenizer.encode(args.prompt)
        if tokenizer.unk_id in ids:
            import sys
            print("提示：prompt 含词表外字符，这些字符将编码为 [UNK]。", file=sys.stderr)
        prompt = torch.tensor([ids], dtype=torch.long, device=device)
        result = generate_tokens(model, prompt, args.max_new_tokens, args.temperature,
                                 args.top_k or None, not args.no_cache)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    # 保留原来的输入，只把新生成的 token 转成文字
    print(args.prompt + tokenizer.decode(result[0, len(ids):].tolist()))


if __name__ == "__main__":
    main()
