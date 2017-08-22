#!/usr/bin/env python
#coding:utf-8
"""
倒立振子
    倒立振子の振り上げ動作の学習
    状態 : 角度、角速度 -->[cos(theta), sine(theta), theta_dot]
    行動 : トルク--> (-1, 1)
    報酬 : 倒立振子が頂点に来た時を原点として現在の状態のなす角を元に以下の式で算出される
        reward = -1.0 * (theta**2 + 0.1 * theta_dot**2 + 0.001*max_toruque**2)
"""

import argparse

import numpy as np
import gym
import sys
from agents.pendulum_agent import Agent

import time

def main(max_episode, gpu, evaluation):
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)

    max_step= 200
    
    success = 0
    failure = 0

    num_state = len(env.observation_space.high)
    num_action = 3
    
    agent = Agent(num_state, num_action, gpu)

    path = "/home/amsl/my_reinforcement_learning_tutorial/models/deep_q_learning/"

    if not evaluation:
        print "Train mode!!!"
    else:
        print "evaluation mode!!!"


    for i_episode in xrange(max_episode):
        state = env.reset()
        for t in xrange(max_step):
            #  env.render()
            #  action = env.action_space.sample()
            action, Q_max = agent.get_action(np.array([state], dtype=np.float32))
            next_state, reward, done, _ = env.step(np.array([action]))

            if not evaluation:
                agent.train(np.array([state], dtype=np.float32), np.array([next_state], dtype=np.float32), np.array([action], dtype=np.int32), np.array([reward], dtype=np.float32), done)

            if done:
                if reward > 0:
                    success += 1
                else:
                    failure += 1

                sys.stdout.write("\repisode : {0:4d} success : {1:4d} failure : {2:4d} success rate : {3:0.3f}".format(i_episode, success, failure, float(success)/float(success+failure)))
                sys.stdout.flush()
                break
            else:
                state = next_state

    print "Success : ", success
    print "Failure : ", failure
    print "Success Rate : ", success / float(max_episode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script is ...')
    parser.add_argument('-i', '--iteration', help='max iteration of episode', type=int, default=1000000)
    parser.add_argument('--gpu', help='gpu mode(0) or cpu mode(-1)', type=int, default=-1)
    parser.add_argument('-e', '--evaluation', help='evaluation mode', type=bool, default=False)

    args = parser.parse_args()
    if args.gpu >= 0:
        print "GPU mode!!"
    else:
        print "CPU mode!!"

    main(args.iteration, args.gpu, args.evaluation)
