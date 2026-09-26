import torch

# 创建一个2*3的全0张量
a = torch.zeros(2,3)
print(a)
print()

# 创建一个全1张量
b = torch.ones(3,4)
print(b)
print()

# 创建一个随机数张量
c = torch.randn(3,4)
print(c)

# 从numpy中搬取数据
import numpy as np
numpy_array = np.array([[1,2],[3,4]])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(tensor_from_numpy)

# 在指定设备上创建张量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = torch.randn(2,3,device=device)
print(d)

