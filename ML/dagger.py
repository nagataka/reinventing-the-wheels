"""
DAgger:
    https://arxiv.org/abs/1011.0686
Utilities are from UC Berkeley CS294 homework.
https://github.com/berkeleydeeprlcourse/homework/tree/master/hw1

Usage:
     python dagger.py experts/Hopper-v1.pkl Models/behavior_cloning.h5 Hopper-v1 --render --num_rollouts 30

"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import keras
import argparse
import random
from decimal import Decimal

TRAIN_RATIO = 0.8
T = 100

def get_model(observations, actions):

    model = keras.Sequential([
        keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(1,11)),
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dense(3,activation='softmax')
    ])

    num_train = int( TRAIN_RATIO*observations.shape[0] )

    X_train = observations[:num_train]
    Y_train = actions[:num_train]
    X_test = observations[num_train:]
    Y_test = actions[num_train:]

    optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

    model.compile(optimizer=optimizer, loss='logcosh', metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=64, nb_epoch=10, verbose=1)
    #model.train_on_batch(X_train, Y_train)

    return model


def main():

    beta = 0.99

    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('agent_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session() as sess:
        tf_util.initialize()

        observations = []
        expert_actions = []
        env = gym.make(args.envname)
        #sum_beta = 0

        pi = keras.models.load_model(args.agent_policy_file)
        print('loaded the agent policy {}'.format(args.agent_policy_file))
        returns = []
        for i in range(args.num_rollouts):
            print('iter', i)
            current_observations = []
            #current_expert_actions = []
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            print("beta is {}".format(beta**i))

            while not done:
                if(random.random() < beta**i):
                    print("Query expert")
                    action = policy_fn(obs[None,:])
                    expert_actions.append(action)
                else:
                    print("Use its own policy")
                    o = np.array([obs.reshape(1,11)])
                    action = pi.predict(o)
                    expert_action = policy_fn(obs[None,:])
                    expert_actions.append(expert_action)

                #print(action.shape)
                current_observations.append(obs.reshape(1,11))
                #actions.append(action.reshape(1,3))
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if (steps % T)==0: 
                    print("{} steps executed. Update policy".format(steps))
                    observations = observations + current_observations
                    pi = get_model(np.array(observations), np.array(expert_actions))
                    current_observations = []


            print("Current rollout has finished. Update policy")
            observations = observations + current_observations
            pi = get_model(np.array(observations), np.array(expert_actions))
            current_observations = []

            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        """
        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        obs = expert_data['observations']
        act = expert_data['actions']
        model = get_model(obs, act)
        num_train = int( TRAIN_RATIO*expert_data['observations'].shape[0] )
        
        X_test = obs[num_train:]
        Y_test = act[num_train:]
        score = model.evaluate(X_test, Y_test, verbose=1)
        print("The score of the model is {}".format(score))
        print(model.metrics_names)
        model.summary()
        model.save("Models/behavior_cloning.h5")
        """


if __name__ == '__main__':
    main()
