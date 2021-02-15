from model import Policy
from simulator import Trainer
from utils import plot_result

import gym
import torch
import argparse


def train(args):
    gamma = args.gamma
    lr = args.lr
    n_episodes = args.n_episodes
    render = args.render
    log_interval = args.log_interval

    trainer = Trainer(
        env=env,
        policy=policy,
        device=device,
        render=render,
        gamma=gamma,
        lr=lr
    )

    return trainer.run(n_episodes, log_interval)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_hidden', type=int, default=128, help="The number of nodes in the hidden layer")
    parser.add_argument('--gamma', type=float, default=0.98, help="Discounted rate for future returns.")
    parser.add_argument('--lr', type=float, default=0.0002, help="Learning rate for updating policy's parameters.")
    parser.add_argument('--n_episodes', type=int, default=10000, help="The number of simulations for policy training.")
    parser.add_argument('--log_interval', type=int, default=100, help="The interval between training status logs.")
    parser.add_argument('--render', type=bool, default=False, help="Whether to render the environment during training, "
                                                                   "Note rendering the environment makes training significantly slower.")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_arguments()

    env = gym.make('CartPole-v1')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = Policy(dim_hidden=args.dim_hidden)
    policy.to(device)

    # Train
    history = train(args)
    plot_result(history, args.log_interval)

    # Test

