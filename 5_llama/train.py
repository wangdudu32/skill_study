"""用文本训练 Mini LLaMA。"""

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from model import MiniLlama, ModelConfig
from tokenizer import CharTokenizer


def sample_batch(data: Tensor, batch_size: int, seq_len: int, generator: torch.Generator,
                 device: str | torch.device) -> tuple[Tensor, Tensor]:
    if data.ndim != 1 or len(data) < seq_len + 1:
        raise ValueError("语料必须是一维 token 序列，长度至少为 seq_len + 1")
    # y 比 x 往后取一个字符，用来预测下一个字符
    starts = torch.randint(len(data) - seq_len, (batch_size,), generator=generator)
    offsets = starts[:, None] + torch.arange(seq_len)[None, :]
    return data[offsets].to(device), data[offsets + 1].to(device)


@torch.inference_mode()
def evaluate(model: MiniLlama, batches: list[tuple[Tensor, Tensor]]) -> float:
    was_training = model.training
    model.eval()
    try:
        return sum(model(x, y).loss.item() for x, y in batches) / len(batches)
    finally:
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser(description="从零训练 Mini LLaMA")
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data/sample.txt")
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("steps", "batch_size", "seq_len", "eval_interval", "eval_batches", "threads"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须是正整数")
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction 必须介于 0 和 1 之间")
    for name in ("lr", "grad_clip"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须是有限正数")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay 必须是有限非负数")
    if args.seq_len > args.max_seq_len:
        parser.error("--seq-len 不能超过 --max-seq-len")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("当前环境未检测到可用 CUDA，请使用 --device cpu")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    try:
        text = args.data.read_text(encoding="utf-8")
        split = int(len(text) * (1 - args.val_fraction))
        train_text, val_text = text[:split], text[split:]
        if min(len(train_text), len(val_text)) <= args.seq_len:
            parser.error("切分后的训练集和验证集都至少需要 seq_len + 1 个字符；请增加语料或减小 --seq-len")
        # 词表只用训练集建，没见过的字符记为 UNK
        tokenizer = CharTokenizer.from_text(train_text)
        train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
        val_data = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
        config = ModelConfig(tokenizer.vocab_size, dim=args.dim, n_layers=args.layers,
                             n_heads=args.heads, n_kv_heads=args.kv_heads,
                             hidden_dim=args.hidden_dim, max_seq_len=args.max_seq_len)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    model = MiniLlama(config).to(device)
    # RMSNorm 的缩放参数不做 weight decay
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": args.weight_decay},
        {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
    ], lr=args.lr)
    # 训练和评估分别采样，评估每次用同一批数据
    train_rng = torch.Generator().manual_seed(args.seed)
    eval_rng = torch.Generator().manual_seed(args.seed + 1)
    eval_sets = {
        name: [sample_batch(data, args.batch_size, args.seq_len, eval_rng, device)
               for _ in range(args.eval_batches)]
        for name, data in (("train", train_data), ("val", val_data))
    }
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"device={device} parameters={num_parameters:,} vocab={tokenizer.vocab_size} "
          f"train_chars={len(train_data)} val_chars={len(val_data)}", flush=True)
    unknown_fraction = (val_data == tokenizer.unk_id).float().mean().item()
    if unknown_fraction:
        print(f"验证集中 {unknown_fraction:.1%} 的字符不在训练词表中，将作为 [UNK] 评估。", flush=True)
    metrics = []
    started = time.perf_counter()

    def record(step: int):
        row = {"step": step, "train_loss": evaluate(model, eval_sets["train"]),
               "val_loss": evaluate(model, eval_sets["val"]),
               "elapsed_seconds": time.perf_counter() - started}
        if not all(math.isfinite(row[key]) for key in ("train_loss", "val_loss")):
            raise RuntimeError("评估 loss 非有限数，请检查数据或减小学习率")
        metrics.append(row)
        print(f"step={step:4d} train_loss={row['train_loss']:.4f} "
              f"val_loss={row['val_loss']:.4f} elapsed={row['elapsed_seconds']:.1f}s", flush=True)

    record(0)
    model.train()
    for step in range(1, args.steps + 1):
        x, y = sample_batch(train_data, args.batch_size, args.seq_len, train_rng, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y).loss
        if not torch.isfinite(loss):
            raise RuntimeError("训练 loss 非有限数，请减小学习率")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            record(step)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / "model.pt"
    # 把配置和词表一起保存，生成时就不用重新读语料了
    torch.save({"format_version": 1, "config": asdict(config), "tokenizer": tokenizer.to_dict(),
                "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "step": args.steps, "metrics": metrics,
                "train_args": {key: str(value) if isinstance(value, Path) else value
                               for key, value in vars(args).items()}}, checkpoint_path)
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已保存：{checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
