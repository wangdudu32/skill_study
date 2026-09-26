import torch

# =====================
# 1. 初始化数据和参数
# =====================

x = torch.tensor([3.0])
y_true = torch.tensor([10.0])

# 模型参数
w = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)

learning_rate = 0.01
target_loss = 1e-4
max_epochs = 1000

# =====================
# 2. 训练循环
# =====================

for epoch in range(max_epochs):

    # 前向传播
    y_pred = w * x + b

    # 均方误差
    loss = (y_pred - y_true) ** 2

    # 判断是否达到目标
    if loss.item() < target_loss:
        print(f"\n损失已达到目标，停止训练")
        break

    # 清空上一轮梯度
    if w.grad is not None:
        w.grad.zero_()
        b.grad.zero_()

    # 反向传播：计算梯度
    loss.backward()

    # 更新参数
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    if epoch % 5 == 0:
        print(
            f"Epoch {epoch:2d} | "
            f"预测值: {y_pred.item():.6f} | "
            f"损失: {loss.item():.6f} | "
            f"w: {w.item():.6f} | "
            f"b: {b.item():.6f}"
        )

# =====================
# 3. 输出最终结果
# =====================

final_pred = w * x + b
final_loss = (final_pred - y_true) ** 2

print("\n最终结果")
print("w =", w.item())
print("b =", b.item())
print("预测值 =", final_pred.item())
print("真实值 =", y_true.item())
print("最终损失 =", final_loss.item())
