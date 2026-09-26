import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN,self).__init__()
        self.fc1 = nn.Linear(2,2)   # 输入层到隐藏层
        self.fc2 = nn.Linear(2,1)   # 隐藏层到输出层

    def forward(self,x):
        x = torch.relu(self.fc1(x))  # 使用ReLU激活函数
        x = self.fc2(x)
        return x

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型实例
model = SimpleNN()

# 将模型移到设备上
model.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()    # 均方差损失函数
optimizer = optim.Adam(model.parameters(),lr=0.001) # Adam优化器

# 假设我们有寻览数据X 和 Y
X = torch.randn(10,2)   # 10 个样本，2个特征
Y = torch.randn(10,1)   # 10 个目标值
# 将数据移到设备上
X = X.to(device)
Y = Y.to(device)

# 训练循环
for epoch in range(100000):    # 训练 100 轮
    optimizer.zero_grad()   # 清空之前的梯度
    output = model(X)   # 前向传播
    loss = criterion(output,Y)  # 计算损失
    loss.backward() # 反向传播
    optimizer.step()    # 更新参数

    if(epoch+1)%100 == 0:
        print(f"Epoch [{epoch+1:6d}/100000] , Loss:{loss.item():.4f}")
