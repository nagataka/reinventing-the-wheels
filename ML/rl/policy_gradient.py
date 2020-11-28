import argparse
import numpy as np

import gym
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.feedforward import Feedforward
from utils.memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

class Agent():
    def __init__(self, algo, optimizer, env, num_actions, memory_size=10000):
        self.algo = algo # currently not used. Future work to make learning algorithms modular.
        self.env = env
        self.policy = Feedforward(env.observation_space.shape[0], env.action_space.n)
        self.memory = ReplayMemory(memory_size)
        self.n_actions = env.action_space.n
        if optimizer == 'Adam':
            self.optim = optim.Adam(self.policy.parameters())
        else:
            raise NotImplementedError

    def update_policy(self, gamma):
        memory = self.memory.get_memory()
        episode_length = len(memory)
        memory = list(zip(*memory))
        discounted_rewards = []
        s = memory[0]
        a = memory[1]
        ns = memory[2]
        r = memory[3]
        a_one_hot = torch.nn.functional.one_hot(torch.tensor(a), num_classes=2).float()

        for t in range(episode_length):
            r_forward = r[t:]
            G = 0
            # Compute a discounted cummulative reward from time step t
            for i, reward in enumerate(r_forward):
                G += gamma**i * reward
            discounted_rewards.append(G)

        rewards_t = torch.tensor(discounted_rewards).detach()
        s_t = torch.tensor(s).float()
        selected_action_probs = self.policy(s_t)
        prob = torch.sum( selected_action_probs*a_one_hot, axis=1  )
        # A small hack to prevent inf when log(0)
        clipped = torch.clamp(prob, min=1e-10, max=1.0)
        J = -torch.log(clipped) * rewards_t
        grad = torch.sum(J)

        self.optim.zero_grad()
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
            t = 0

            while not done:
                action_prob = self.policy( torch.from_numpy(state).float() )
                #print(action_prob.detach().numpy())
                #action = torch.argmax(action_prob).item()
                action = np.random.choice(range(self.n_actions), p=action_prob.detach().numpy())
                next_state, reward, done, _ = self.env.step(action)
                self.memory.push(state, action, next_state, reward)
                self.env.render()

                total_reward += reward
                state = next_state 
                t += 1

            print("Episode: ", e)
            print("Reward: ", total_reward)
            writer.add_scalar('./runs/rewards', total_reward, e)
            
            self.update_policy(gamma)
        self.env.close()
        writer.close()


def main():
    parser = argparse.ArgumentParser(description="Main for training RL agent")
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1,000)')
    parser.add_argument('--optimizer', type=str, default='Adam', help="optimizer")
    parser.add_argument('--env', type=str, default='CartPole-v0', help="https://github.com/openai/gym/wiki/Table-of-environments")
    parser.add_argument('--algo', type=str, default='reinforce', help="learning algorithm")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env_name = args.env

    env = gym.make(env_name)
    num_actions = env.action_space.n # or shape[0]
    num_observations = env.observation_space.shape[0]
    print("Created {} env which has {} actions in {} spaces".format(env_name, num_actions, num_observations) )

    agent = Agent(args.algo, args.optimizer, env, num_actions)
    agent.train(args.epochs, args.gamma)

if __name__ == "__main__":
    main()
