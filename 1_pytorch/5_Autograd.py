import torch

x = torch.randn(2,3,requires_grad=True)
print(x)

y = x + 2
print(f"y:{y}")

z = y*y*3
print(f"z:{z}")

out = z.mean()
print(out)

print("\n\n反向传播后")
out.backward()
print(f"x.grad:{x.grad}")
