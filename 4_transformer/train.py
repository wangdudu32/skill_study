"""训练数字反转任务。"""

import argparse
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import PAD_ID, VOCAB_SIZE, ReverseDataset, collate_batch, decode_numbers
from transformer import Transformer, TransformerConfig, greedy_decode


def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    for src, decoder_input, labels in loader:
        src, decoder_input, labels = (x.to(device) for x in (src, decoder_input, labels))
        optimizer.zero_grad(set_to_none=True)
        logits = model(src, decoder_input)
        loss_sum = criterion(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1))
        token_count = (labels != PAD_ID).sum().item()
        # 只按非 PAD 的 token 数算平均 loss
        loss = loss_sum / token_count
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss_sum.item()
        total_tokens += token_count
    return total_loss / total_tokens


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")
    total_loss = 0.0
    correct_tokens = total_tokens = correct_sequences = total_sequences = 0
    for src, decoder_input, labels in loader:
        src, decoder_input, labels = (x.to(device) for x in (src, decoder_input, labels))
        logits = model(src, decoder_input)
        total_loss += criterion(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1)).item()
        valid = labels != PAD_ID
        correct_tokens += ((logits.argmax(dim=-1) == labels) & valid).sum().item()
        total_tokens += valid.sum().item()

        # 上面用的是真实前缀，这里让模型自己一步步生成
        # 整条序列都对才算正确
        generated = greedy_decode(model, src, max_new_tokens=labels.size(1))[:, 1:]
        predictions = torch.full_like(labels, PAD_ID)
        predictions[:, : generated.size(1)] = generated
        correct_sequences += (predictions == labels).all(dim=1).sum().item()
        total_sequences += labels.size(0)
    return {
        "loss": total_loss / total_tokens,
        "token_accuracy": correct_tokens / total_tokens,
        "sequence_accuracy": correct_sequences / total_sequences,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从零训练 Transformer，学习数字序列反转")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2, help="Encoder 和 Decoder 各自的层数")
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=1, help="CPU 线程数；小矩阵通常无需很多线程")
    parser.add_argument("--checkpoint", default="checkpoints/reverse.pt")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    for name in ("epochs", "train_size", "val_size", "batch_size", "threads", "learning_rate"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} 必须大于 0")
    if not 1 <= args.min_length <= args.max_length:
        parser.error("需要满足 1 <= min-length <= max-length")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("当前环境没有可用的 CUDA，请使用 --device cpu")
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(device_name)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.threads)

    try:
        config = TransformerConfig(
            src_vocab_size=VOCAB_SIZE,
            tgt_vocab_size=VOCAB_SIZE,
            d_model=args.d_model,
            n_heads=args.heads,
            num_encoder_layers=args.layers,
            num_decoder_layers=args.layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            max_len=args.max_length + 1,  # 给 EOS/BOS 多留一个位置
            pad_id=PAD_ID,
        )
    except ValueError as error:
        parser.error(str(error))
    train_data = ReverseDataset(args.train_size, args.min_length, args.max_length, args.seed)
    val_data = ReverseDataset(args.val_size, args.min_length, args.max_length, args.seed + 1)
    # 数据打乱单独用一个随机数生成器
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_batch, generator=generator,
    )
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_batch)
    model = Transformer(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"设备：{device} | 参数量：{parameter_count:,} | 训练/验证：{len(train_data)}/{len(val_data)}", flush=True)
    best_score = (-1.0, float("-inf"))
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        # 先看整条序列的准确率，一样时再比较 loss
        score = (metrics["sequence_accuracy"], -metrics["loss"])
        improved = score > best_score
        if improved:
            best_score = score
            torch.save({
                "model_config": asdict(config),
                "model_state_dict": model.state_dict(),
                "train_config": vars(args),
                "epoch": epoch,
                "metrics": metrics,
            }, checkpoint_path)
        print(
            f"epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} "
            f"| val_loss={metrics['loss']:.4f} "
            f"| token_acc={metrics['token_accuracy']:.2%} "
            f"| sequence_acc={metrics['sequence_accuracy']:.2%}"
            f"{' | 已保存' if improved else ''}",
            flush=True,
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    elapsed = time.perf_counter() - started
    print(f"\n耗时 {elapsed:.1f}s，最佳模型：{checkpoint_path}（epoch {checkpoint['epoch']}）")
    samples = [val_data[index] for index in range(min(3, len(val_data)))]
    src, _, labels = collate_batch(samples)
    predictions = greedy_decode(model, src.to(device), max_new_tokens=config.max_len).cpu()
    for source, target, prediction in zip(src.tolist(), labels.tolist(), predictions.tolist()):
        print(f"输入：{decode_numbers(source)} | 目标：{decode_numbers(target)} | 预测：{decode_numbers(prediction)}")


if __name__ == "__main__":
    main()
