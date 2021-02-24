import torch

from torch.optim import Adam
from torch.distributions import Categorical


class REINFORCEAgent:
    def __init__(self, env, policy, lr, gamma, mode='train', render=False):
        self.env = env

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = policy.to(self.device)
        self.gamma = gamma
        self.render = render

        self.mode = mode
        if self.mode == 'train':
            self.policy.train()
            self.optimizer = Adam(self.policy.parameters(), lr=lr)
        else:
            self.policy.eval()

        self.data = []

    def select_action(self, s):
        probs = self.policy.forward(s)
        m = Categorical(probs)
        a = m.sample()
        return a, probs[a]

    def run_episode(self):
        # Reset the initial state
        s = self.env.reset()
        if self.render:
            self.env.render()
        done = False

        score = 0.0
        while not done:
            s = torch.FloatTensor(s).to(self.device)
            a, p = self.select_action(s)

            # Apply action to environment
            s, r, done, info = self.env.step(a.item())
            if self.render:
                self.env.render()
            self.data.append((r, p))
            score += r

        if self.mode == 'train':
            self.train()

        self.data = []
        return score

    def train(self):
        self.policy.train()
        self.optimizer.zero_grad()

        R = 0
        loss = []
        for r, p in self.data[::-1]:
            R = r + self.gamma * R
            loss.append(R * torch.log(p))
        loss = -1 * torch.mean(torch.stack(loss))

        loss.backward()
        self.optimizer.step()
