import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, dim_hidden):
        super(Policy, self).__init__()

        self.fc1 = nn.Linear(4, dim_hidden)  # CartPole-v0 has four dimensional state space
        self.fc2 = nn.Linear(dim_hidden, 2)  # CartPole-v0 has two actions (left + 1, right + 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=0)
        return x
