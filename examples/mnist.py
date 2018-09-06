import torch
from torch import nn
from torch.nn import functional as F

import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST

import flare
from flare import Trainer

# Data loading and transformation

train_data = MNIST('/Users/Abhai/datasets/', train=True, download=True)
test_data = MNIST('/Users/Abhai/datasets/', train=False, download=True)

train_X, train_Y = train_data.train_data, train_data.train_labels
test_X, test_Y = test_data.test_data, test_data.test_labels

# Conv layers require 4D inputs
train_X = torch.unsqueeze(train_X, 1).float()
test_X = torch.unsqueeze(test_X, 1).float()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net()

t = Trainer(model, nn.NLLLoss(), torch.optim.Adam(model.parameters()))
t.train(train_X, train_Y, validation_split=0.2, batch_size=128)