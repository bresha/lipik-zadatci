import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import DataLoader
from moje_mreze import MyNet
from torch import nn
from torch import optim
import torch

transform = transforms.Compose([Resize((28, 28)), ToTensor()])

LR = 1e-3
BATCH_SIZE = 16
NUM_OF_EPOCHS = 10


device = 'cuda' if torch.cuda.is_available() else ' cpu'

mnist_dataset_train = MNIST(root='data', train=True, transform=transform, download=True)
mnsit_dataset_val = MNIST(root='data', train=False, transform=transform, download=True)

sample = mnist_dataset_train[0]

print(mnist_dataset_train[0][0].shape)

# fig, axes = plt.subplots(3)
# for i in range(3):
#     image, label = mnist_dataset_train[i]
#     image = image.squeeze()
#     axes[i].set_title(str(label))
#     axes[i].imshow(image, cmap='gray')
# plt.show()

mnist_dataloader_train = DataLoader(dataset=mnist_dataset_train, batch_size=BATCH_SIZE, shuffle=True)
mnsit_dataloader_val = DataLoader(dataset=mnsit_dataset_val, batch_size=BATCH_SIZE, shuffle=True)


model = MyNet().to(device=device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

def train(train_dataloader, model, loss_fn, optim, current_epoch):
    dataset_size = len(train_dataloader.dataset)
    running_loss = 0

    for batch_iter, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)

        predictions = model(images)

        loss = loss_fn(predictions, labels)

        optim.zero_grad()

        loss.backward()

        optim.step()

        running_loss += loss.item()
        if batch_iter % 100 == 0:
            average_loss = running_loss / 100
            current_itereation = batch_iter * len(labels)
            print('[%d] [%d / %d] loss: %3f' % (current_epoch, current_itereation, dataset_size, average_loss))
            running_loss = 0


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print("Accuracy: " + str(100 * correct) + "%, Average loss: " + str(test_loss))


def validate(val_dataloader, model):
    correct = 0
    total = 0
    all_preds = torch.tensor([])

    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy: ', 100 * correct // total)



for epoch in range(NUM_OF_EPOCHS):
    train(mnist_dataloader_train, model, loss_fn, optimizer, epoch)
    test(mnsit_dataloader_val, model, loss_fn)

torch.save(model, 'model.pt')
