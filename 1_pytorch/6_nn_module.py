import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN,self).__init__()
        self.fc1 = nn.Linear(2,2)
        self.fc2 = nn.Linear(2,1)
    
    def forward(self,x):
        x = torch.relu(self.fc1(x)) # Relu激活函数
        x = self.fc2(x)
        return x

# 创建神经网络实例
model = SimpleNN()

# 打印模型结构
# print(model)

x = torch.randn(1,2)

# 前向传播
output = model(x)
print(f"output:{output}")

# 定义损失函数（例如:均方差损失函数）
criterion = nn.MSELoss()

# 随机设置一个目标值
target = torch.randn(1,1)
print(f"target:{target}")

# 计算损失
loss = criterion(output,target)
print(f"loss:{loss}")

# 定义优化器
optimizer = optim.Adam(model.parameters(),lr = 0.01)

# 训练步骤
optimizer.zero_grad()	# 清空上一轮的梯度
loss.backward()	# 反向传播，计算梯度
optimizer.step()	# 更新参数



















