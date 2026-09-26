import torch

x = torch.tensor([3.0])
y_true = torch.tensor([10.0])

learning_rate = 0.001
max_epoches = 1000
target_loss = 1e-110

# 权重和偏置值
w = torch.tensor([2.0],requires_grad = True)
b = torch.tensor([1.0],requires_grad = True)

# 训练循环
for epoch in range(max_epoches):
    y_pred = w*x + b
    # 前向传播
    loss = (y_pred - y_true)**2

    # 判断损失是否已经降低到目标值
    if loss.item() < target_loss:
        print(f"target_loss : {target_loss}")
        print(f"loss : {loss.item()}")
        print("\n损失已经降到目标值以下，训练完成")
        break

    # 反向传播之前需要先清空上一轮的梯度(因为反向传播的梯度是采用累加的方式)
    if w.grad is not None:
        w.grad.zero_()
        b.grad.zero_()

    # 反向传播，计算梯度
    loss.backward()
    
    # 更新参数
    with torch.no_grad():
        w -= learning_rate*w.grad
        b -= learning_rate*b.grad

    if epoch % 5 == 0 :
        print(f"Epoch : {epoch:2d} | "
                f"预测值:{y_pred.item():.6f} |"
                f"损失值:{loss.item():.6f} |"
                f"w : {w.item():.6f} |"
                f"b : {b.item():.6f}"
        )
    
# 输出最终损失
final_pred = w*x + b
final_loss = (final_pred - y_true)**2

print("\n最终结果")
print(f"w:{w.item():.6f}")
print(f"b:{b.item():.6f}")
print(f"预测值:{final_pred.item():.6f}")
print(f"真实值:{y_true.item()}")
print(f"最终损失:{final_loss.item():.6f}")

