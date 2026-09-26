import torch

a = torch.tensor([[1,2],[3,4]])
b = torch.tensor([[5,6],[7,8]])

print(a,'\n')
print(b,'\n')

# 逐个元素（对应位置的元素）相加
print(a+b,'\n')

# 逐个元素（对应位置的元素）相乘
print(a*b,'\n')

# 张量的转置
c = torch.ones(2,4)
print(c,'\n')
print("转置之后")
print(c.t(),'\n')

# 张量的形状
print(c.shape)
