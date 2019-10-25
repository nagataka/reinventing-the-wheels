import argparse
import numpy as np

import gym
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.feedforward import Feedforward, ValueNetwork
from utils.memory import ReplayMemory
#import utils.util
from tensorboardX import SummaryWriter

def main():
    parser = argparse.ArgumentParser(description="Main for training RL agent")
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1,000)')
    parser.add_argument('--optimizer', type=str, default='Adam', help="optimizer")
    parser.add_argument('--env', type=str, default='CartPole-v0', help="https://github.com/openai/gym/wiki/Table-of-environments")
    parser.add_argument('--algo', type=str, default='dqn', help="learning algorithm")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env_name = args.env

    env = gym.make(env_name)
    num_actions = env.action_space.n # or shape[0]
    num_observations = env.observation_space.shape[0]
    print("Created {} env which has {} actions in {} spaces".format(env_name, num_actions, num_observations) )

    algo = import_module('algorithms.'+args.algo)
    agent = Agent(algo, args.optimizer, env, num_actions)
    agent.train(args.epochs, args.gamma)

writer = SummaryWriter()

class Agent():
    def __init__(self, algo, optimizer, env, num_actions, memory_size=10000):
        self.algo = algo # currently not used. Future work to make learning algorithms modular.
        self.env = env
        self.policy = Feedforward(env.observation_space.shape[0], env.action_space.n)
        self.value_network = ValueNetwork(env.observation_space.shape[0], 1)
        self.optim_value = optim.Adam(self.value_network.parameters())
        self.memory = ReplayMemory(memory_size)
        if optimizer == 'Adam':
            self.optim_policy = optim.Adam(self.policy.parameters())

    def update(self, gamma):
        memory = self.memory.get_memory()
        discounted_rewards = []
        J = []
        errors = []
        for t in range(len(memory)):
            G = 0
            state, reward = memory[t][0], memory[t][4]
            state_tensor = torch.from_numpy(state).float()
            v_st = self.value_network( state_tensor )

            trajectory = memory[t:]

            k = t+1
            for transition in trajectory:
                # Compute a discounted cummulative reward from time step t
                G += gamma**t * transition[4]
                k += 1
            delta = G - v_st
            discounted_rewards.append(G)
            J.append( -transition[2]*delta)
            #print("Total rewards with discounting from time {} is {} ".format(t, G))

            v_st = self.value_network( state_tensor )
            error = (v_st - reward)**2
            errors.append(error)

        # Update V
        self.value_network.zero_grad()
        # https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#baselines-in-policy-gradients
        phi = torch.mean( torch.stack(errors) )
        phi.backward()
        self.optim_value.step()

        self.policy.zero_grad()
        grad = torch.sum(torch.stack(J))
        grad.backward()
        self.optim_policy.step()
            
        return phi


    def train(self, num_epochs, gamma):

        for e in range(num_epochs):
            # initialization
            self.memory.flush()
            state = self.env.reset()
            done = False
            total_reward = 0
            #self.env.render()

            while not done:
                state_tensor = torch.from_numpy(state).float()
                action_prob = F.softmax(self.policy( state_tensor ))
                action = torch.argmax(action_prob).item()
                log_prob = torch.log( action_prob[action] )
                next_state, reward, done, _ = self.env.step(action)
                self.memory.push(state, action, log_prob, next_state, reward)
                #self.env.render()

                total_reward += reward
                state = next_state 

            value_estimate_error = self.update(gamma)

            print("Done with reward ", total_reward)
            writer.add_scalar('./runs/value_estimate', value_estimate_error, e)
            writer.add_scalar('./runs/rewards', total_reward, e)
            
        self.env.close()
        writer.close()

if __name__ == "__main__":
    main()
