import argparse
import numpy as np
import random
from collections import namedtuple

import gym
from gym import wrappers

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

UPDATE_CYCLE = 100  # frequency to update agent
CEM_PERCENTILE = 70 # threshold of 'elite' episode
R = 1               # action repeat

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['state', 'action'])

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions, bound):
        super(Net, self).__init__()
        self.bound = bound
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
            nn.Tanh()
        )
        print(self.net)

    def forward(self, x):
        return self.net(x)*self.bound[1]


class Agent():
    def __init__(self, env, net, n_actions, device):
        self.env = env
        self.net = net
        self.num_actions = n_actions
        self.criterion = nn.MSELoss()
        #self.criterion = nn.CrossEntropyLoss() # if the action space is discrete
        self.optimizer = optim.Adam(params=self.net.parameters())
        self.device = device
        self.sm = nn.Softmax()
        self.writer = SummaryWriter(comment=self.env.unwrapped.spec.id)
        self.best_mean_reward = -float('inf')

    def update(self, batch, e):
        rewards = list(map(lambda s: s.reward, batch))
        reward_bound = np.percentile(rewards, CEM_PERCENTILE)
        reward_mean = float(np.mean(rewards))
        print("Reward bound: ", reward_bound)
        print("Reward mean: ", reward_mean)
        states = []
        actions = []

        n = 0
        for episode in batch:
            if episode.reward <= reward_bound:
                continue
            n += 1
            states.extend(map(lambda step: torch.tensor(step.state, dtype=torch.float, device=self.device), episode.steps))
            actions.extend(map(lambda step: torch.tensor(step.action, dtype=torch.float, device=self.device), episode.steps))
            #actions.extend(map(lambda step: torch.tensor(step.action, dtype=torch.long, device=self.device), episode.steps)) # Use this with CrossEntropyLoss
        if n == 0:
            print("No elite episode(s)")
            return None
        else:
            print("Update policy with {} elites".format(n))
            state_tensors = torch.tensor((len(states), self.env.observation_space.shape[0]), dtype=torch.float, device=self.device)
            action_tensors = torch.tensor((len(actions), self.env.action_space.shape[0]), dtype=torch.float, device=self.device)
            #action_tensors = torch.tensor((len(actions), self.env.action_space.shape[0]), dtype=torch.long, device=self.device) # Use this with CrossEntropyLoss
            state_tensors = torch.cat(states, out=state_tensors).view(len(states), self.env.observation_space.shape[0])
            action_tensors = torch.cat(actions, out=action_tensors).view(len(actions), self.env.action_space.shape[0])
        self.optimizer.zero_grad()

        """ Use this with discrete action space and CrossEntropyLoss()
        action_probs = torch.div( torch.add(self.net(state_tensors),2), 4)
        action_tensors = torch.div( torch.add(action_tensors,2), 4)
        loss = self.criterion(action_probs, action_tensors.view(-1))
        """
        action_probs = self.net(state_tensors)
        # Compute MSE between what your current policy tells you is good and what worked well in the past rollout
        loss = self.criterion(action_probs, action_tensors)

        print("Loss: ", loss)
        loss.backward()
        self.optimizer.step()
        if reward_mean > self.best_mean_reward:
            torch.save(self.net.state_dict(), 'CEM-' + self.env.unwrapped.spec.id + ".pt")
            print("Best mean reward updated %.3f -> %.3f, model saved" % (self.best_mean_reward, reward_mean))
            self.best_mean_reward = reward_mean
        print("Best mean reward so far: ", self.best_mean_reward)
        self.writer.add_scalar("loss", loss.item(), e)
        self.writer.add_scalar("reward_bound", reward_bound, e)
        self.writer.add_scalar("reward_mean", reward_mean, e)

    def train(self, num_epochs):

        batch = []
        for e in range(num_epochs):
            # initialization
            state = self.env.reset()
            done = False
            total_reward = 0.0
            episode_steps = []

            while not done:
                #self.env.render()
                state_tensor = torch.tensor(state, dtype=torch.float, device=self.device)
                action = self.net(state_tensor).item()
                noise = np.random.normal(0, 0.3, (self.num_actions,)) # Inject noise for exploration
                action = np.clip( action + noise, self.env.action_space.low, self.env.action_space.high ).astype(np.float32)
                # Repeat the action R times (Experimental. Didn't see improvement yet and currently set R=1)
                for r in range(R):
                    next_state, reward, done, _ = self.env.step(action)
                    episode_steps.append(EpisodeStep(state=state, action=action))

                    total_reward += reward
                    state = next_state 

            batch.append(Episode(reward=total_reward, steps=episode_steps))
            episode_steps = []
            if (e%UPDATE_CYCLE == 0) and (e != 0):
                print("Update parameters")
                self.update(batch, e)
                batch = []
        
        self.env.close()

    def test(self, num_epochs):
        for e in range(num_epochs):
            # initialization
            state = self.env.reset()
            done = False
            total_reward = 0.0
            maximum_reward = -float('inf')
            reward_history = []
        
            while not done:
                self.env.render()
                state_tensor = torch.tensor(state, dtype=torch.float, device=self.device)
                action = self.net(state_tensor).item()

                for r in range(R):
                    next_state, reward, done, _ = self.env.step([action])
                    total_reward += reward
                    state = next_state

            print("Done with reward ", total_reward)
            reward_history.append(total_reward)
            if total_reward > maximum_reward:
                maximum_reward = total_reward

        self.env.close()
        return maximum_reward, reward_history

def main():
    parser = argparse.ArgumentParser(description="Main for training RL agent")
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--epochs', type=int, default=10000000, metavar='N',
                        help='number of epochs to train (default: 10,000)')
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = gym.make('Pendulum-v0')
    env_str = env.unwrapped.spec.id 

    if isinstance(env.action_space, gym.spaces.box.Box):
        num_actions = env.action_space.shape[0]
    else:
        num_actions = env.action_space.n
    num_observations = env.observation_space.shape
    print("Created {} env which has {} actions in {} spaces".format(env_str, num_actions, num_observations) )
    print('  - low:', env.action_space.low.item())
    print('  - high:', env.action_space.high.item())
    input_shape = num_observations[0]
    net = Net(input_shape, hidden_size=128, n_actions=num_actions, bound=(env.action_space.low.item(), env.action_space.high.item())).to(device)
    if args.test:
        assert args.model_path != None, "Specify a model path"
        net.load_state_dict(torch.load(args.model_path))
        agent = Agent(env, net, num_actions, device)
        maximum_reward, reward_history = agent.test(args.epochs)
    else:
        agent = Agent(env, net, num_actions, device)
        agent.train(args.epochs)
    agent.writer.close()

if __name__ == "__main__":
    main()
