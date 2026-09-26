# 手撕 Mini LLaMA

用 PyTorch 写一个小型 LLaMA，主要是为了学习模型结构。包含 RMSNorm、RoPE、GQA、SwiGLU 和 KV cache，代码里加了中文注释。

使用字符级 tokenizer，可以在示例文本上训练，然后给一段开头让模型续写。这是从头训练的小模型，没有加载官方权重。

## 环境

Python 3.10 及以上，依赖是 PyTorch。

```bash
python3 -m pip install torch
```

如果用项目里已有的虚拟环境，先运行 `source .venv_llama/bin/activate`。

## 文件说明

```text
model.py             模型实现
tokenizer.py         字符和 token ID 之间的转换
train.py             训练模型
generate.py          加载模型并生成文本
data/sample.txt      示例语料
tests/               测试代码
```

## 运行

先用示例语料训练：

```bash
python3 train.py
```

默认训练 200 步，模型保存到 `out/model.pt`，训练记录保存在 `out/metrics.json`。有可用的 CUDA 就用 GPU，否则用 CPU。

训练好后，给一段文字让模型接着写：

```bash
python3 generate.py --prompt '春天来了，' --temperature 0
```

`--temperature 0` 表示每次选概率最大的字符。想试试随机采样，可以这样运行：

```bash
python3 generate.py --prompt '春天来了，' --temperature 0.8 --top-k 20
```

默认开启 KV cache，加上 `--no-cache` 可以关闭。模型按指定长度生成，不会自动在句末停下。

换成自己的 UTF-8 文本：

```bash
python3 train.py --data data/my_text.txt --out-dir out/my_model
python3 generate.py --checkpoint out/my_model/model.pt --prompt '你的开头'
```

运行测试：

```bash
python3 -m unittest discover -s tests -v
```

其他参数可以看 `python3 train.py --help` 和 `python3 generate.py --help`。

示例语料比较少，生成结果容易重复，主要用来跑通训练和生成流程。

结构参考：[Llama 3 模型代码](https://github.com/meta-llama/llama3/blob/main/llama/model.py)。
