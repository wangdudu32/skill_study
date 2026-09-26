"""生成数字反转数据。"""

import random
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
DIGIT_OFFSET = 3
VOCAB_SIZE = 13  # 3 个特殊符号 + 10 个数字


def encode_numbers(numbers: Sequence[int]) -> list[int]:
    """数字加 3 得到 token ID，前 3 个 ID 留给特殊符号。"""
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
    """右侧补 PAD，输入和标签错开一位。

    比如目标是 [BOS, a, b, EOS]，输入取 [BOS, a, b]，标签取 [a, b, EOS]。
    """
    src_length = max(len(src) for src, _ in batch)
    tgt_length = max(len(tgt) for _, tgt in batch)
    src_batch = torch.full((len(batch), src_length), PAD_ID, dtype=torch.long)
    tgt_batch = torch.full((len(batch), tgt_length), PAD_ID, dtype=torch.long)
    for row, (src, tgt) in enumerate(batch):
        src_batch[row, : len(src)] = torch.tensor(src)
        tgt_batch[row, : len(tgt)] = torch.tensor(tgt)
    return src_batch, tgt_batch[:, :-1], tgt_batch[:, 1:]
