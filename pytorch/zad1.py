from moje_mreze import MyNet
import torch
import cv2
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else ' cpu'

model = torch.load('model.pt').to(device)
model.eval()

print(summary(model, (1, 28, 28)))

orig_img = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)

scaled_img = cv2.resize(orig_img, dsize=(28, 28))

img = np.reshape(scaled_img, (1, 28, 28))

img = img.astype('float32')

img = torch.tensor(img)


img = torch.unsqueeze(img, 0)


img = img.to(device)

print(img.shape)


result = model(img)
print(result)
softmax = torch.nn.Softmax(dim=-1)
result = softmax(result)
print(result)

print('result: ', result.argmax(1).item())

cv2.imshow('image', scaled_img)
cv2.waitKey(0)
cv2.destroyAllWindows()