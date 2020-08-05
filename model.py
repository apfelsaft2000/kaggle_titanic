import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self,input_size):
        super(CNN,self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2,2)

        self.conv1 = nn.Conv1d(1,3,3)
        #self.conv2 = nn.Conv1d(3,3,2)

        self.fc1 = nn.Linear(6,120)
        self.fc2 = nn.Linear(120,2)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        #x = self.conv2(x)
        #x = self.relu(x)
        #x = self.pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
