import argparse
import numpy as np
import copy

import gym
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.feedforward import GaussianConvPolicy
from utils.memory import ReplayMemory
import utils.util as util
from tensorboardX import SummaryWriter

def main():
    parser = argparse.ArgumentParser(description="Main for training RL agent")
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1,000)')
    parser.add_argument('--optimizer', type=str, default='Adam', help="optimizer")
    parser.add_argument('--env', type=str, default='CarRacing-v0', help="https://github.com/openai/gym/wiki/Table-of-environments")
    parser.add_argument('--algo', type=str, default='policy_gradient', help="learning algorithm")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env_name = args.env

    env = gym.make(env_name)
    num_actions = env.action_space.shape[0]
    num_observations = env.observation_space.shape[0]
    print("Created {} env which has {} actions in {} spaces".format(env_name, num_actions, num_observations) )

    algo = import_module('algorithms.'+args.algo)
    agent = Agent(algo, args.optimizer, env, num_actions, device)
    agent.train(args.epochs, args.gamma)

writer = SummaryWriter()

class Agent():
    def __init__(self, algo, optimizer, env, num_actions, device, memory_size=10000):
        self.algo = algo # currently not used. Future work to make learning algorithms modular.
        self.env = env
        self.device = device
        self.policy = GaussianConvPolicy(3, env.action_space.shape[0]).to(device)
        self.memory = ReplayMemory(memory_size)
        if optimizer == 'Adam':
            self.optim = optim.Adam(self.policy.parameters())

    def update_policy(self, gamma):
        memory = self.memory.get_memory()
        discounted_rewards = []
        J_t = []
        for t in range(len(memory)):
            G = 0
            trajectory = memory[t:]
            k = t+1
            for transition in trajectory:
                # Compute a discounted cummulative reward from time step t
                G += gamma**k * transition[4]
                k += 1
            discounted_rewards.append(G)
            J_t.append( -transition[2] * G)
            #print("Total rewards with discounting from time {} is {} ".format(t, G))
        self.policy.zero_grad()
        grad = torch.sum(torch.stack(J_t))
        grad.backward()
        self.optim.step()


    def train(self, num_epochs, gamma):
        # initialization

        for e in range(num_epochs):
            self.memory.flush()
            state = self.env.reset()
            done = False
            total_reward = 0
            self.env.render()

            while not done:
                current_state_copy = copy.deepcopy(state)
                action_tensor, log_prob = self.policy( util.img2tensor(current_state_copy, self.device) )
                # action: steer, gas, brake
                action = [ action_tensor[0][0].item(), action_tensor[0][1].item(), action_tensor[0][2].item() ]
                next_state, reward, done, _ = self.env.step(action)
                self.memory.push(state, action, log_prob, next_state, reward)
                self.env.render()

                total_reward += reward
                state = next_state 

            print("Done with reward ", total_reward)
            writer.add_scalar('./runs/rewards', total_reward, e)
            
            self.update_policy(gamma)
        self.env.close()
        writer.close()

if __name__ == "__main__":
    main()
