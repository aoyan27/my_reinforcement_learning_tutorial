#!/usr/bin/env python
#coding:utf-8
"""
Frozen lake
    4x4('FrozenLake-v0') or 8x8('FrozenLake8x8-v0')の2つの大きさの環境が用意されている
    状態 : マップ上の自身の位置が数字として与えられる(Q_tableはマップ全体に対して確保される)
    行動 : 上下左右の4パターン(確率で選択した方向から+-1ずれることあり、例えば、上を選択すると左右も動作候補として考慮され1/3の確率で選択される)
    報酬 : ゴールに辿り着いたら報酬-->2 それ以外-->0
"""

import argparse

import numpy as np
import gym
from agents.frozen_lake_agent import Agent
import sys
import time


def main(max_episode, alpha, gamma, evoluation):
    env_name = "FrozenLake-v0"
    env = gym.make(env_name)

    agent = Agent(env, alpha, gamma)
    
    #  max_episode = 10000000
    max_step = 100
    
    success_times = 0
    failure_times = 0

    path = "/home/amsl/my_reinforcement_learning_tutorial/models/q_learning/"
    
    if not evoluation:
        print "Train mode!!!"
    else:
        print "Evoluation mode!!!"

    if evoluation:
        load_path = path + env_name + "_max_episode-" + str(max_episode) + ".npy"
        agent.load_model(load_path)
        agent.display_q_table()

    count = 0

    for i_episode in xrange(max_episode):
        #  print "episode : {}".format(i_episode+1)
        sys.stdout.flush()
        #  time.sleep(0.01)
        state = env.reset()
        for t in xrange(max_step):
            #  env.render()
            action = env.action_space.sample()
            #  agent.epsilon_decay(count)
            count += 1

            action = agent.get_action(state, args.evoluation)
            next_state, reward, done, info = env.step(action)
            #  print "state : {0}, action : {1}, next_state : {2}, reward : {3}, done : {4}, info : {5}".format(state, action, next_state, reward, done, info)
            #  sys.stdout.write("\repisode : {0}, state : {1}, action : {2}, next_state : {3}, reward : {4}, done : {5}, info : {6}".format(i_episode, state, action, next_state, reward, done, info))
            #  sys.stdout.write("\repisode : {0:5d}, success : {1:5d} failure : {2:5d}".format(i_episode, success_times, failure_times))
            #  sys.stdout.flush()
            #  time.sleep(0.01)
            if evoluation:
                agent.q_update(state, next_state, action, reward, done)

            if done:
                #  print "Episode finishde after {} time steps.".format(t+1)
                if reward > 0:
                    success_times += 1
                else:
                    failure_times += 1
                sys.stdout.write("\repisode : {0:5d} epsilon : {4:0.3f} success : {1:5d} failure : {2:5d} success rate : {3:0.3f}".format(i_episode, success_times, failure_times, float(success_times)/float(success_times+failure_times), agent.epsilon))

                break
            else:
                state = next_state
    agent.display_q_table()
    print "Success : ", success_times
    print "Failure : ", failure_times
    print "Success Rate : ", success_times / float(max_episode)
    if evoluation:
        file_name = env_name + "_max_episode-" + str(max_episode) + ".npy"
        abs_path = path + file_name
        print "Save model ---> {}".format(abs_path)
        agent.save_model(abs_path)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script is ...')
    parser.add_argument('-i', '--iteration', help='max iteration of episode',
                        type=int, default=1000000)
    parser.add_argument('-a', '--alpha', help='alpha(learning rate)',
                        type=float, default=0.1)
    parser.add_argument('-g', '--gamma', help='gamma(discount rate)',
                        type=float, default=0.99)
    parser.add_argument('-e', '--evoluation', help='evoluation mode',
                        type=bool, default=False)

    args = parser.parse_args()
    main(args.iteration, args.alpha, args.gamma, args.evoluation)
