import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from collections import namedtuple


class ActorCriticAgent:
    def __init__(self, env, actor, critic, lr, gamma, mode='train', render=False):
        self.env = env

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        self.gamma = gamma
        self.render = render

        # Set model train (or eval)
        self.mode = mode
        if self.mode == 'train':
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)
            self.actor.train()
            self.critic.train()

        else:
            self.actor.eval()
            self.critic.eval()

        self.Transition = namedtuple('Transition', ('s', 'a', 'r', 's_prime', 'done'))
        self.transitions = []

    def select_action(self, state):
        """
        :param state: np.ndarray with shape [dim_state, 1]
        :return action: python scalar
        """
        # Convert "state" to torch.tensor and make shape [1, dim_state]
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.actor(state)  # "probs" shape: [1, dim_action]
        m = Categorical(probs)

        # Since "probs" is a torch.tensor, so "action" is. The method ".item()" converts torch.tensor to python scalar
        action = m.sample().item()
        return action

    def train(self):
        # Make torch.tensor from the transitions
        transitions = self.Transition(*zip(*self.transitions))
        s = torch.tensor(transitions.s, dtype=torch.float32, device=self.device)  # shape: [len(current episode), dim_state]
        a = torch.tensor(transitions.a, device=self.device).unsqueeze(1)  # shape: [len(current episode, 1]
        r = torch.tensor(transitions.r, device=self.device).unsqueeze(1)  # shape: [len(current episode, 1]
        s_prime = torch.tensor(transitions.s_prime, dtype=torch.float32, device=self.device)  # # shape: [len(current episode), dim_state]
        done_mask = list(map(lambda x: 0.0 if x else 1.0, transitions.done))
        done = torch.tensor(done_mask, device=self.device).unsqueeze(1)  # shape: [len(current episode), dim_state]

        td_target = r + self.gamma * self.critic(s_prime) * done
        delta = td_target - self.critic(s)

        # "probs" shape [len(current episode), dim_action], probabilities of all possible actions given the state at each time
        probs = self.actor(s)
        probs_a = probs.gather(1, a)  # "probs_a" shape [len(the episode), 1] : probabilities of actions selected at each time
        loss = -torch.log(probs_a) * delta.detach() + F.smooth_l1_loss(self.critic(s), td_target.detach())

        # Update the parameters
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        loss.mean().backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()

    def run_episode(self):
        s = self.env.reset()  # "s" (initial state), np.ndarray with shape [4, 1]
        if self.render:
            self.env.render()
        done = False

        score = 0.0
        while not done:
            a = self.select_action(s)  # Sample an action
            s_prime, r, done, _ = self.env.step(a)  # Interact with the environment
            if self.render:
                self.env.render()
            self.transitions.append(self.Transition(s, a, r / 100.0, s_prime, done))

            score += r
            s = s_prime

        if self.mode == "train":
            self.train()

        self.transitions = []
        return score
