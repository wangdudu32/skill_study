import torch

w = torch.tensor([2.0],requires_grad = True)
b = torch.tensor([1.], requires_grad = True)
# 一条训练数据
x = torch.tensor([3.])
y_true = torch.tensor([10.])

y_pred = w*x + b

# 损失函数
# 前向计算
L = (y_pred - y_true)**2

# 反向传播
L.backward()

print("预测值:",y_pred.item())
print("实际值:",y_true)
print("w的梯度:",w.grad)
print("b的梯度:",b.grad)

