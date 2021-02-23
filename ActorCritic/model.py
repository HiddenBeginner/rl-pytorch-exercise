import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, dim_state, dim_hidden, dim_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(dim_state, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_action)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


class Critic(nn.Module):
    def __init__(self, dim_state, dim_hidden):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(dim_state, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
