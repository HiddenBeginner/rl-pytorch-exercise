import numpy as np
import matplotlib.pyplot as plt


def plot_result(history, interval):
    history = np.array(history)
    x = interval * np.arange(0, len(history))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, history,'b-', label='Reward')
    ax.set_xlabel('Episode', fontsize=15)
    ax.set_ylabel('Reward', fontsize=15)
    ax.set_title('The average rewards of last 100 episodes', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.legend()
    plt.show()
