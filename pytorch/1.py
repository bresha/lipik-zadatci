import torch
print(torch.cuda.is_available())
import numpy as np

x = torch.rand(5, 3)
print(x)

python_polje = [1, 2, 3, 4]
tensor = torch.tensor(python_polje)
print(tensor)

np_polje = np.array(python_polje)
tensor_is_np = torch.tensor(np_polje)
print(tensor_is_np)

tensor_a = torch.ones_like(tensor)
print(tensor_a)

tensor_b = torch.rand_like(tensor, dtype=torch.float)
print(tensor_b)


rand_tensor = torch.rand(size=(2,2))
ones_tensor = torch.ones(size=(3,1))
zeros_tensor = torch.zeros(size=(2,3))

print(zeros_tensor)

print(zeros_tensor.shape)

if torch.cuda.is_available():
    tensor = tensor.to('cuda')

print(tensor.device)

a = torch.rand(size=(2, 3))

print(a[0])
print(a[:, 1])
print(a[:, 2])


a = torch.tensor([[1, 2],
                    [3, 4],
                    [5, 6]])


b = torch.tensor([[1, 2, 3],
                    [4, 5, 6]])


c = a.matmul(b)

print(c)

a = torch.tensor([23])

b = a.item()

print(b)

a = torch.tensor([[[1, 2],
                  [3, 4]]])

b = torch.squeeze(a, 0)

print(b.shape)

a = torch.tensor([[1, 2],
                    [3, 4],
                    [5, 6]])

b = a.unsqueeze(0)

print(b)


a = torch.randn(10, 10)
a = a.view(100)
print(a)

print('done')

a = a.view(-1, 20)
print(a)

