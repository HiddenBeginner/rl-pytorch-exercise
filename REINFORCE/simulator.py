import time
import torch

from torch.optim import Adam
from torch.distributions import Categorical


class Simulator:
    def __init__(self, env, policy, device, render=False):
        self.env = env
        self.policy = policy
        self.device = device
        self.render = render

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

        data = []
        score = 0.0
        while not done:
            s = torch.FloatTensor(s).to(self.device)
            a, p = self.select_action(s)

            # Apply action to environment
            s, r, done, info = self.env.step(a.item())
            if self.render:
                self.env.render()
            data.append((r, p))
            score += r

        return data, score


class Trainer(Simulator):
    def __init__(self, env, policy, device, render, gamma, lr):
        super(Trainer, self).__init__(env, policy, device, render)
        self.gamma = gamma
        self.optimizer = Adam(self.policy.parameters(), lr=lr)

    def finish_episode(self, data):
        self.policy.train()
        self.optimizer.zero_grad()

        R = 0
        loss = []
        for r, p in data[::-1]:
            R = r + self.gamma * R
            loss.append(R * torch.log(p))
        loss = -1 * torch.mean(torch.stack(loss))

        loss.backward()
        self.optimizer.step()

    def run(self, n_episodes, log_interval):
        now = time.time()
        scores = 0.0
        history = []
        self.policy.train()
        for i in range(n_episodes):
            data, score = self.run_episode()
            self.finish_episode(data)
            scores += score
            if (i + 1) % log_interval == 0:
                print(f"[Episode {i + 1}] Average reward of last {log_interval} episodes: {scores / log_interval} | " +
                      f"Elapsed time (sec): {time.time() - now:.3f} |", end='\r')
                history.append(scores / log_interval)
                scores = 0.0
        return history

    def save(self):
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        })
