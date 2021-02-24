from model import Policy
from agent import REINFORCEAgent
from utils import plot_result

import gym
import argparse


def main(args):
    env = gym.make('CartPole-v0')
    policy = Policy(dim_hidden=args.dim_hidden)
    agent = REINFORCEAgent(env=env, policy=policy, lr=args.lr, gamma=args.gamma, render=args.render)

    scores = 0
    history = []
    for i in range(args.n_episodes):
        scores += agent.run_episode()
        if (i + 1) % args.print_interval == 0:
            print(f"[Episode {i + 1}] Avg Score: {scores / args.print_interval:.3f}")
            history.append(scores / args.print_interval)
            scores = 0.0

    plot_result(history, args.print_interval)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_hidden', type=int, default=128, help="The number of nodes in the hidden layer")
    parser.add_argument('--gamma', type=float, default=0.98, help="Discounted rate for future returns.")
    parser.add_argument('--lr', type=float, default=0.00015, help="Learning rate for updating policy's parameters.")
    parser.add_argument('--n_episodes', type=int, default=10000, help="The number of simulations for policy training.")
    parser.add_argument('--print_interval', type=int, default=100, help="The interval between training status logs.")
    parser.add_argument('--render', type=int, default=0, help="Whether to render the environment during training, "
                                                              "Note rendering the environment makes training significantly slower.")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_arguments()
    main(args)