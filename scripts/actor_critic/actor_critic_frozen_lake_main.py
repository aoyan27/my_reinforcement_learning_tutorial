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
import sys
import copy

from agents.actor_critic_frozen_lake_agent import Agent

import time


def main(max_episode, alpha, gamma, evaluation):
    env_name = "FrozenLake-v0"
    env = gym.make(env_name)

    agent = Agent(env, alpha, gamma)

    max_step = 100

    success = 0
    failure = 0

    path = "/home/amsl/my_reinforcement_learning_tutorial/models/actor_critic/"

    if not evaluation:
        print "Train mode!!!"
    else:
        print "evaluation mode!!!"


    for i_episode in xrange(max_episode):
        state = env.reset()
        for t in xrange(max_step):
            #  env.render()

            #  action = env.action_space.sample()
            action = agent.actor(state, evaluation)
            #  print "action : ", action

            next_state, reward, episode_end, info = env.step(action)
            
            if not evaluation:
                agent.train(state, action, next_state, reward, episode_end)
            #  print "\repisode : {0:5d}, state : {1:2d},  action : {2:1d}, next_state : {3:2d}, reward : {4:0.1f}, episode_end : {5}.".format(i_episode, state, action, next_state, reward, episode_end)
            

            #  state = copy.deepcopy(next_state)
            state = next_state

            #  print "agent._mu_table : ", agent._mu_table
            #  print "agent._sigma_table : ", agent._sigma_table
            #  print "agent._v_table : ", agent._v_table

            if episode_end:
                if reward > 0:
                    success += 1
                else:
                    failure += 1
                break

        #  sys.stdout.write("\repisode : {0:5d}, state : {1:2d},  action : {2:1d}, next_state : {3:2d}, reward : {4:0.1f}, episode_end : {5}, success : {6:5d}, failure : {7:5d}, success_rate : {8:0.3f}.".format(i_episode, state, action, next_state, reward, episode_end, success, failure, float(success)/float(success+failure)))
        #  sys.stdout.flush()
        print "\repisode : {0:5d}, state : {1:2d},  action : {2:1d}, next_state : {3:2d}, reward : {4:0.1f}, episode_end : {5}, success : {6:5d}, failure : {7:5d}, success_rate : {8:0.3f}.".format(i_episode, state, action, next_state, reward, episode_end, success, failure, float(success)/float(success+failure))
        print "agent._mu_table : ", agent._mu_table
        print "agent._sigma_table : ", agent._sigma_table
        print "agent._v_table : ", agent._v_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script is ...')
    parser.add_argument('-i', '--iteration', help='max iteration of episode',
                        type=int, default=1000000)
    parser.add_argument('-a', '--alpha', help='alpha(learning rate)',
                        type=float, default=0.1)
    parser.add_argument('-g', '--gamma', help='gamma(discount rate)',
                        type=float, default=0.99)
    parser.add_argument('-e', '--evaluation', help='evaluation mode',
                        type=bool, default=False)

    args = parser.parse_args()
    main(args.iteration, args.alpha, args.gamma, args.evaluation)
