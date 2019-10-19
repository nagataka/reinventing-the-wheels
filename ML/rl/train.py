import argparse
import numpy as np

import gym
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.cnn import CNN
from utils.memory import ReplayMemory

def main():
    parser = argparse.ArgumentParser(description="Main for training RL agent")
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1,000)')
    parser.add_argument('--env', type=str, default=None, help="https://github.com/openai/gym/wiki/Table-of-environments")
    parser.add_argument('--algo', type=str, default='dqn', help="learning algorithm")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env_str = args.env

    env = gym.make(env_str)
    num_actions = env.action_space.shape[0] # if the type is Discrete, then .n
    num_observations = env.observation_space
    print("Created {} env which has {} actions in {} spaces".format(env_str, num_actions, num_observations) )

    algo = import_module('algorithms.'+args.algo)
    agent = Agent(algo, env, num_actions)
    agent.train(args.epochs)


class Agent():
    def __init__(self, algo, env, num_actions, memory_size=10000):
        self.algo = algo
        self.env = env
        self.policy = CNN(3, 96, 96, 1)
        self.memory = ReplayMemory(memory_size)

    def train(self, num_epochs):
        # initialization
        state = self.env.reset()
        done = False
        total_reward = 0
        self.env.render()

        for e in range(num_epochs):
            while not done:
                action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)
                self.env.render()

                total_reward += reward
                state = next_state 

            print("Done with reward ", total_reward)

if __name__ == "__main__":
    main()
