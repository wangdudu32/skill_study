# 从零手写 Mini Llama

配套代码是 `mini_llama.py`，按文件顺序阅读即可。这是一个保留 Llama 3 核心结构的小模型，参数随机初始化，目的是理解结构、训练和自回归生成。示例使用人为构造的 token 编号，不包含正式 tokenizer 或预训练权重。

参考：[Meta 的 Llama 3 实现](https://github.com/meta-llama/llama3/blob/main/llama/model.py)。

## 运行

项目使用现有 `.venv`。需要重新安装依赖时，可以使用：

```bash
uv pip install --python .venv/bin/python --index https://download.pytorch.org/whl/cpu 'torch==2.8.0+cpu'
uv pip install --python .venv/bin/python numpy
.venv/bin/python mini_llama.py
```

## 第一步：固定形状的含义

| 符号 | 含义 | 示例 |
| --- | --- | --- |
| B | batch size，序列条数 | 8 |
| T | sequence length，每条序列的 token 数 | 16 |
| D | hidden dimension，每个 token 的向量维度 | 64 |
| V | vocabulary size，词表大小 | 32 |
| H | 查询头数 | 4 |
| H_kv | K/V 头数 | 2 |
| d | 每个头的维度，D / H | 16 |

数据流：

```text
token_ids [B,T]
    → Embedding [B,T,D]
    → 多层 TransformerBlock [B,T,D]
    → RMSNorm [B,T,D]
    → lm_head [B,T,V]
```

Embedding 是一张 `[V,D]` 的可训练表，token 编号用于查表。输出 `logits[b,t,:]` 表示看到位置 `t` 及之前的 token 后，对下一个 token 的预测分数。

## 第二步：RMSNorm

对每个 token 的 D 个特征计算平方均值，再除以它的平方根：

```text
mean_square = mean(x², 最后一维)
y = x / sqrt(mean_square + eps) * weight
```

`weight` 是形状为 `[D]` 的可学习缩放参数，初始化为全 1。Llama 使用的 RMSNorm 不减均值，也没有可学习的偏置。

例如 `x=[3,4]`，平方均值是 `12.5`，忽略 eps、且 weight 为全 1 时，输出约为 `[0.8485,1.1314]`。输出的均方根约为 1，均值并不一定是 0。

`mean(dim=-1, keepdim=True)` 把 `[B,T,D]` 变成 `[B,T,1]`，这样每个 token 都能除以自己的归一化因子。计算时转成 float32，降低低精度计算平方与求和时的数值误差。

## 第三步：先理解注意力，再读 RoPE

同一个输入 x，通过三个独立线性层得到 Q、K、V：

```text
Q = Wq(x)：当前位置用什么特征查找信息
K = Wk(x)：每个位置用什么特征供查询匹配
V = Wv(x)：匹配后实际汇总的信息
```

一个头的计算是：

```text
scores = Q @ Kᵀ / sqrt(d)
probabilities = softmax(scores + mask, dim=-1)
output = probabilities @ V
```

Q、K 的点积衡量匹配程度。除以 `sqrt(d)` 有助于控制分数尺度。softmax 沿着被读取的位置计算，让每一行成为一组权重，随后对 V 加权求和。

## 第四步：RoPE 给 Q、K 加上位置信息

RoPE 不增加向量维度。它把每个头内的特征两两配对，对每一对做二维旋转。

第 i 对特征在位置 p 的旋转角度为：

```text
frequency_i = theta^(-2i / d)
angle = p * frequency_i
```

设一对特征是 `(a,b)`，旋转后为：

```text
a_new = a*cos(angle) - b*sin(angle)
b_new = a*sin(angle) + b*cos(angle)
```

本实现采用相邻配对：`(x0,x1), (x2,x3), ...`。对 Q、K 应用相同频率的旋转后，它们的点积会体现相对位置差。这一步作用于 Q 和 K，V 保留原值。

代码里 `0::2` 提取偶数索引，`1::2` 提取奇数索引，`stack(...).flatten(-2)` 再把旋转后的特征交错还原。相邻配对与其他实现可能使用的半区配对，需要和对应的权重布局保持一致，不能在加载权重时直接混用。

## 第五步：GQA 与因果 mask

GQA 的 Q 头数可以多于 K/V 头数。例如 4 个 Q 头共享 2 个 K/V 头：

```text
Q0、Q1 → K0、V0
Q2、Q3 → K1、V1
```

形状依次变化：

```text
Q: [B,T,D] → [B,T,H,d]    → [B,H,T,d]
K: [B,T,D] → [B,T,H_kv,d] → [B,H_kv,T,d]
V: [B,T,D] → [B,T,H_kv,d] → [B,H_kv,T,d]
```

代码使用 `repeat_interleave` 把 K/V 头按组重复到 H 个，方便显式做矩阵乘法。投影参数仍然由每组查询头共享；优化实现可以避免物理重复 K/V。

`Q @ K.transpose(-2,-1)` 得到 `[B,H,T,T]`，倒数第二维是发起查询的位置，最后一维是被读取的位置。

模型预测下一个 token 时，只能读取当前位置及其之前的信息，因此在 softmax 之前屏蔽未来位置：

```text
长度为 4 时的加性 mask：
 0   -inf -inf -inf
 0     0  -inf -inf
 0     0    0  -inf
 0     0    0    0
```

`exp(-inf)=0`，因此这些位置的注意力权重为 0。训练时所有位置可以并行计算，而因果约束依然成立。

多个头的结果为 `[B,H,T,d]`，转置并合并后得到 `[B,T,D]`，最后经过输出投影 `wo`。

## 第六步：SwiGLU 前馈网络

```text
gate = SiLU(W_gate(x))
up = W_up(x)
output = W_down(gate * up)
```

这里的 `*` 是逐元素相乘。两条分支先从 D 维扩展到 hidden_dim，其中 gate 分支调节另一条分支的特征，再映射回 D 维。

注意力负责跨 token 汇总信息，前馈网络在每个 token 内变换特征。代码中的 hidden_dim=192 是教学配置，并非某个官方模型的完整配置。

## 第七步：组装 TransformerBlock 和完整模型

每层包含两次 Pre-Norm 和两次残差连接：

```python
x = x + attention(attn_norm(x))
x = x + feed_forward(ffn_norm(x))
```

两行中的 x 都保持 `[B,T,D]`，因此可以逐元素相加。第二行使用第一行更新后的 x。

完整模型先查 Embedding，再经过多个 Block，最后通过 RMSNorm 和线性层生成 `[B,T,V]` 的 logits。

## 第八步：训练目标必须错开一位

```text
完整序列：[0,1,2,3,4]
模型输入：[0,1,2,3]
训练目标：[1,2,3,4]
```

实现中：

```python
inputs = sequence[:, :-1]
targets = sequence[:, 1:]
logits = model(inputs)
loss = F.cross_entropy(
    logits.reshape(-1, vocab_size),
    targets.reshape(-1),
)
```

这里的 cross_entropy 需要原始 logits，内部会处理 log-softmax。`reshape` 把 B 和 T 合并，相当于对 B*T 个位置分别计算分类损失后求平均。

接着依次清空梯度、反向传播、更新参数：

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

示例学习循环规律 `0→1→…→7→0`。loss 下降并能延续这个规律，说明前向、损失和反向传播链路能够工作；这不代表模型已经学会自然语言。

## 第九步：自回归生成

```python
logits = model(token_ids)[:, -1, :]
next_token = logits.argmax(dim=-1, keepdim=True)
token_ids = torch.cat((token_ids, next_token), dim=1)
```

每次只使用最后一个位置的分数，选出下一个 token，再接到输入末尾继续预测。示例默认贪心选择；temperature 大于 0 时，代码从 `softmax(logits / temperature)` 中采样。

## 后续优化：KV Cache

当前生成函数每轮重新计算完整前缀，方便理解。KV Cache 可以在保持因果注意力结果的前提下，复用历史 token 的 K/V：

1. 每一层分别保存该层历史的 K 和 V，缓存的是应用 RoPE 后的 K。
2. 第一次处理完整提示词，之后每轮只输入新生成的 token。
3. 新 Q/K 的 RoPE 位置从历史长度开始，不能每轮都从 0 开始。
4. 把当前 K/V 拼到历史 K/V 后面；GQA 缓存使用原始的 H_kv 个头。
5. 当前 Q 读取历史及当前的 K/V。分块输入时，需要根据绝对位置构造矩形因果 mask，不能直接沿用从左上角开始的方阵下三角 mask。

这属于推理优化，本文件尚未实现缓存。先掌握这里的完整前向计算，再加入缓存，更容易通过比较两者 logits 检查正确性。
