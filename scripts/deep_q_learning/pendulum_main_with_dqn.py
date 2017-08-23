#!/usr/bin/env python
#coding:utf-8
"""
倒立振子
    倒立振子の振り上げ動作の学習
    状態 : 角度、角速度 -->[cos(theta), sine(theta), theta_dot]
    行動 : トルク--> (-1, 1)
    報酬 : 倒立振子が頂点に来た時を原点として現在の状態のなす角を元に以下の式で算出される
        reward = -1.0 * (theta**2 + 0.1 * theta_dot**2 + 0.001*max_toruque**2)
    DQNを参考にしている(Experience Replay, Fixed Q-Network, Reward Clipping)
"""

import argparse

import numpy as np
import gym
import sys
from agents.pendulum_agent_with_dqn import Agent

import time

def main(max_episode, gpu, evaluation):
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    #  env = gym.wrappers.Monitor(env, '/tmp/pendulum-experiment-1')

    max_step= 200
    
    success = 0
    failure = 0

    num_state = len(env.observation_space.high)
    num_action = 2

    agent = Agent(num_state, num_action, gpu)
    

    path = "/home/amsl/my_reinforcement_learning_tutorial/models/deep_q_learning/"

    if gpu >= 0:
        print "GPU mode!!"
    else:
        print "CPU mode!!"

    if not evaluation:
        print "Train mode!!!"
    else:
        print "evaluation mode!!!"
    
    t = 0

    for i_episode in xrange(max_episode):
        state = env.reset()
        for j_step in xrange(max_step):
            if i_episode%10000 == 0:
                env.render()
            
            state = np.array([state], dtype=np.float32)
            
            #  action = env.action_space.sample()
            action, _ = agent.get_action(state)
            agent.reduce_epsilon()
            next_state, reward, done, _ = env.step(action)
            next_state = np.array([next_state], dtype=np.float32)

            agent.stock_experience(t, state, action, next_state, reward, done)


            if not evaluation:
                if t > agent.init_exprolation:
                    agent.extract_replay_memory()
                    agent.train()

            agent.target_model_update(t)
            t += 1

            if done:
                if reward > 0:
                    success += 1
                else:
                    failure += 1

                sys.stdout.write("\repisode : {0:4d} success : {1:4d} failure : {2:4d} success rate : {3:0.3f} epsilon : {4:0.6f} t : {5}".format(i_episode, success, failure, float(success)/float(success+failure), agent.epsilon, t -1))
                sys.stdout.flush()
                break
            else:
                state = next_state

    print "Success : ", success
    print "Failure : ", failure
    print "Success Rate : ", success / float(max_episode)

    agent.save_model(path+"dqn_pendulum.model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script is ...')
    parser.add_argument('-i', '--iteration', help='max iteration of episode', type=int, default=1000000)
    parser.add_argument('--gpu', help='gpu mode(0) or cpu mode(-1)', type=int, default=-1)
    parser.add_argument('-e', '--evaluation', help='evaluation mode', type=bool, default=False)

    args = parser.parse_args()

    main(args.iteration, args.gpu, args.evaluation)
