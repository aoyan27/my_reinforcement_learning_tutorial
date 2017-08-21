#!/usr/bin/env python
#coding: utf-8
"""
倒立振子
    倒立振子の振り上げ動作の学習
    状態 : 角度、角速度 -->[cos(theta), sine(theta), theta_dot]
    行動 : トルク--> (-1, 1)
    報酬 : 倒立振子が頂点に来た時を原点として現在の状態のなす角を元に以下の式で算出される
        reward = theta**2 + 0.1 * theta_dot**2 + 0.001*max_toruque**2
"""

import numpy as np

class Agent:
    ALPHA = 0.5
    GAMMA = 0.9
    

    def __init__(self, env, num_action, ):
        #  self.__q_table = np.zeros((), dtype=np.float32)
        print "env.action_space : ", env.action_space
        print "env.action_space.high : ", env.action_space.high
        print "env.action_space.low : ", env.action_space.low
        print "env.observation_space : ", env.observation_space
        print "env.observation_space.high : ", env.observation_space.high
        print "env.observation_space.low : ", env.observation_space.low
        self.__epsilon = 0.1
        

if __name__=="__main__":
    import gym
    env = gym.make('Pendulum-v0')
    num_action = len(np.arange(-1, 2, 2))
    print np.arange(-1, 2, 2)
    test_agent = Agent(env, num_action)
