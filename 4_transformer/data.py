"""数字序列反转任务：例如输入 1 2 3，期望输出 3 2 1。"""

import random
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
DIGIT_OFFSET = 3
VOCAB_SIZE = 13  # PAD、BOS、EOS，加上数字 0～9。


def encode_numbers(numbers: Sequence[int]) -> list[int]:
    """数字与 token ID 不是一回事：数字 0 对应 token 3，不是 PAD。"""
    if not numbers or any(number < 0 or number > 9 for number in numbers):
        raise ValueError("输入必须是非空数字序列，每个数字在 0～9 之间")
    return [number + DIGIT_OFFSET for number in numbers]


def decode_numbers(tokens: Sequence[int]) -> list[int]:
    numbers = []
    for token in tokens:
        if token == EOS_ID:
            break
        if token not in (PAD_ID, BOS_ID):
            numbers.append(token - DIGIT_OFFSET)
    return numbers


class ReverseDataset(Dataset):
    def __init__(self, size: int, min_length: int = 3, max_length: int = 8, seed: int = 42) -> None:
        if size <= 0 or not 1 <= min_length <= max_length:
            raise ValueError("size 必须为正，且 1 <= min_length <= max_length")
        rng = random.Random(seed)
        self.samples = []
        for _ in range(size):
            length = rng.randint(min_length, max_length)
            tokens = encode_numbers([rng.randrange(10) for _ in range(length)])
            src = tokens + [EOS_ID]
            tgt = [BOS_ID] + list(reversed(tokens)) + [EOS_ID]
            self.samples.append((src, tgt))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.samples[index]


def collate_batch(batch: list[tuple[list[int], list[int]]]) -> tuple[Tensor, Tensor, Tensor]:
    """右侧补 PAD，再切出错开一位的 Decoder 输入和监督标签。

    完整目标：[BOS, 3, 2, 1, EOS]
    Decoder 输入：[BOS, 3, 2, 1]，标签：[3, 2, 1, EOS]。
    这里的 1/2/3 表示数字，实际存储的是它们对应的 token ID。
    """
    src_length = max(len(src) for src, _ in batch)
    tgt_length = max(len(tgt) for _, tgt in batch)
    src_batch = torch.full((len(batch), src_length), PAD_ID, dtype=torch.long)
    tgt_batch = torch.full((len(batch), tgt_length), PAD_ID, dtype=torch.long)
    for row, (src, tgt) in enumerate(batch):
        src_batch[row, : len(src)] = torch.tensor(src)
        tgt_batch[row, : len(tgt)] = torch.tensor(tgt)
    return src_batch, tgt_batch[:, :-1], tgt_batch[:, 1:]
