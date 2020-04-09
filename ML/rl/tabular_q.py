import argparse
import random
import math
import gym
import numpy as np
import matplotlib.pyplot as plt
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
#ENV_NAME = "FrozenLake8x8-v0"
GAMMA = 0.9
ALPHA = 0.2
NUM_EPOCHS = 2000
visualize_frequency = 50 # specify NUM_EPOCHS if one figure is enough

# A parameter for controlling simple epsilon-greedy action selection
eps = 0.1

# Parameters for controlling decaying epsilon-greedy action selection
eps_start = 0.9
eps_end = 0.05
eps_decay = 400

class TQ_learner:
    def __init__(self, is_slippery=True):
        self.env = gym.make(ENV_NAME, is_slippery=is_slippery)
        self.num_actions = self.env.action_space.n
        self.num_states = self.env.observation_space.n
        self.state = self.env.reset()
        self.values = np.zeros((self.num_states, self.num_actions)) # action-value table
        self.visit_counter = collections.defaultdict()
        self.total_steps = 0
        self.total_reward = 0

    def update_Q(self, curr_state, action, next_state, reward):
        """
        """
        target = reward + GAMMA * np.max(self.values[next_state])
        diff = target - self.values[curr_state][action]
        self.values[curr_state][action] = self.values[curr_state][action] + ALPHA * diff

    def select_action_greedy(self, state):
        """Return an action given current state. Action to take is determined greedily based on Q (state-action) value.
        """
        action_values = self.values[state]
        max_value = np.max(action_values)
        idx = np.where(action_values == max_value)[0]
        if len(idx) > 1:
            # if multiple actions have the same high value, randomly pick one from those actions
            return np.random.choice(idx, 1)[0]
        else:
            return np.argmax(action_values)

    def select_action_epsgreedy(self, state):
        """Return an action given current state. 
        Explore with probability of epsilon. When to exploit, action is determined greedily based on Q (state-action) value.
        Exploration probability is fixed.

        """
        # Generates a random value [0.0, 1.0) uniformly and perform greedy action selection if the value is greater than eps
        # Else, take a random action (= esplore with probability eps)
        if random.random() > eps:
            action_values = self.values[state]
            max_value = np.max(action_values)
            idx = np.where(action_values == max_value)[0]
            if len(idx) > 1:
                # if multiple actions have the same high value, randomly pick one from those actions
                return np.random.choice(idx, 1)[0]
            else:
                return np.argmax(action_values)
        else:
            return self.env.action_space.sample()

    def select_action_decaying_epsgreedy(self, state):
        """Return an action given current state. 
        Explore with probability epsilon. When to exploit, action is determined greedily based on Q (state-action) value.
        Exploration probability is decaying as more steps taken.
        """
        # Generates a random value [0.0, 1.0) uniformly and perform greedy action selection if the value is greater than eps
        # Else, take a random action (= esplore with probability eps)
        eps = eps_end + (eps_start - eps_end) * math.exp(-1. * self.total_steps / eps_decay)
        if random.random() > eps:
            action_values = self.values[state]
            max_value = np.max(action_values)
            idx = np.where(action_values == max_value)[0]
            if len(idx) > 1:
                # if multiple actions have the same high value, randomly pick one from those actions
                return np.random.choice(idx, 1)[0]
            else:
                return np.argmax(action_values)
        else:
            return self.env.action_space.sample()

    def train(self, exp):
        writer = SummaryWriter(comment="-FrozenLake-Q")
        for epoch in range(NUM_EPOCHS):
        
            self.state = self.env.reset()
            epoch_reward = 0
            done = False

            while not done:
                if exp == 'greedy':
                    action = self.select_action_greedy(self.state)
                elif exp == 'eps':
                    action = self.select_action_epsgreedy(self.state)
                elif exp == 'decay-eps':
                    action = self.select_action_decaying_epsgreedy(self.state)

                obs, reward, done, info = self.env.step(action)
                #self.env.render()
                self.update_Q(self.state, action, obs, reward)
                self.state = obs
                epoch_reward += reward
                self.total_steps += 1
    
                if done:
                    print("Episode {} finished with reward {}".format(epoch, epoch_reward))
                    self.total_reward += epoch_reward
                    if (epoch != 0) and (epoch%20 == 0):
                        val = self.total_reward / 20.
                        print("val: ", val)
                        writer.add_scalar("reward", val, epoch)
                        self.total_reward = 0
            """
            if epoch % visualize_frequency == 0:
                self.visualize_values()
            """
        writer.close()

    def visualize_values(self):
        V = np.max(self.values, axis=1).reshape((int(np.sqrt(self.num_states)), int(np.sqrt(self.num_states))))
        plt.imshow(V, cmap='plasma')
        plt.colorbar()
        plt.show()
        

def main():
    parser = argparse.ArgumentParser(description="Main for training RL agent")
    parser.add_argument('--exp', type=str, default='greedy', help="exploration scheme")
    parser.add_argument('--noslip', type=bool, default=False, help="make the env not slippery")
    args = parser.parse_args()

    if args.noslip:
        agent = TQ_learner(is_slippery=False)
    else:
        agent = TQ_learner()
    assert args.exp in ['greedy', 'eps', 'decay-eps'], "Specified exploration scheme doesn't exist!"
    agent.train(args.exp)
    print("Finished {} epochs. Result: {} total rewards.".format(NUM_EPOCHS, agent.total_reward))
    agent.visualize_values()

if __name__ == "__main__":
    main()
