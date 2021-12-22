import numpy as np

a = np.array([
    [1, 2, 8, 4],
    [2, 6, 4, 5]
])

print(a.min())
print(a.argmin())

print(a.sum())
print(a.mean())
print(a.prod())

a.sort(0)
print(a)
