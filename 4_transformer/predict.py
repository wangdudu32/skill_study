"""加载模型，预测反转后的数字序列。"""

import argparse
from pathlib import Path

import torch

from data import EOS_ID, VOCAB_SIZE, decode_numbers, encode_numbers
from transformer import Transformer, TransformerConfig, greedy_decode


def main() -> None:
    parser = argparse.ArgumentParser(description="使用训练好的 Transformer 反转数字序列")
    parser.add_argument("--checkpoint", default="checkpoints/reverse.pt")
    parser.add_argument("--numbers", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("threads 必须大于 0")
    if not Path(args.checkpoint).is_file():
        parser.error(f"找不到模型 {args.checkpoint}，请先运行 python3 train.py")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("当前环境没有可用的 CUDA，请使用 --device cpu")
    try:
        src_tokens = encode_numbers(args.numbers) + [EOS_ID]
    except ValueError as error:
        parser.error(str(error))
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(device_name)
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = TransformerConfig(**checkpoint["model_config"])
    if config.src_vocab_size != VOCAB_SIZE or config.tgt_vocab_size != VOCAB_SIZE:
        parser.error("此推理脚本只支持数字序列反转任务的词表")
    if len(src_tokens) > config.max_len:
        parser.error(f"此模型最多接受 {config.max_len - 1} 个数字")
    model = Transformer(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    src = torch.tensor([src_tokens], dtype=torch.long, device=device)
    # 数字个数不变，再留一个位置给 EOS
    output = greedy_decode(model, src, max_new_tokens=len(src_tokens))[0].tolist()
    print(f"输入：{args.numbers}")
    print(f"预测：{decode_numbers(output)}")
    if EOS_ID not in output:
        print("模型达到生成长度上限，尚未生成 EOS。")


if __name__ == "__main__":
    main()
