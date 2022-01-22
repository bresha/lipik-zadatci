from torch import nn
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5, 5), padding=2)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(5, 5), padding=2)
        self.fc1 = nn.Linear(in_features=20*7*7, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 980) # -1 is batch size
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)

        self.pool = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)

        self.conv5 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv6 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv7 = nn.Conv2d(256, 256, (3, 3), padding=1)

        self.conv8 = nn.Conv2d(256, 512, (3, 3), padding=1)
        self.conv9 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv10 = nn.Conv2d(512, 512, (3, 3), padding=1)

        self.conv11 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv12 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv13 = nn.Conv2d(512, 512, (3, 3), padding=1)

        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv8(x)
        x = F.relu(x)
        x = self.conv9(x)
        x = F.relu(x)
        x = self.conv10(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv11(x)
        x = F.relu(x)
        x = self.conv12(x)
        x = F.relu(x)
        x = self.conv13(x)
        x = F.relu(x)
        # x = self.pool(x)

        x = x.view(-1, 512)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x
