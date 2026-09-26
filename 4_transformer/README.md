# 手撕 Transformer

用 PyTorch 从头实现一个 Transformer，主要是为了学习它的结构。包含 Encoder 和 Decoder，注意力、位置编码和 LayerNorm 都自己写了一遍，代码里加了中文注释。

用数字序列反转做了个简单的训练例子，比如输入 `1 2 3 4`，输出 `4 3 2 1`。数据由程序生成，不用另外下载。

## 环境

Python 3.10 及以上，依赖是 PyTorch。

```bash
python3 -m pip install -r requirements.txt
```

## 文件说明

```text
transformer.py    模型实现
data.py           生成数据和整理 batch
train.py          训练模型
predict.py        加载模型并预测
tests/            测试代码
```

## 运行

先训练模型：

```bash
python3 train.py
```

默认训练 20 轮，模型保存到 `checkpoints/reverse.pt`。有可用的 CUDA 就用 GPU，否则用 CPU。

训练好后试一下预测：

```bash
python3 predict.py --numbers 1 2 3 4
```

输出例子：

```text
输入：[1, 2, 3, 4]
预测：[4, 3, 2, 1]
```

默认训练的序列长度是 3～8，每个数字在 0～9 之间。想调整训练参数可以看 `python3 train.py --help`。

运行测试：

```bash
python3 -m unittest discover -s tests -v
```
