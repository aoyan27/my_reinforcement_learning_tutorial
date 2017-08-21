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

import numpy as np
import gym
from agents.pendulum_agent import Agent

def main():
    env = gym.make('Pendulum-v0')
    #  print "env.action_space : ", env.action_space
    #  print "env.action_space.high : ", env.action_space.high
    #  print "env.action_space.low : ", env.action_space.low
    #  print "env.observation_space : ", env.observation_space
    #  print "env.observation_space.high : ", env.observation_space.high
    #  print "env.observation_space.low : ", env.observation_space.low
    num_state = len(env.observation_space.high)
    num_action = 2
    agent = Agent(num_state, num_action)

    max_episode = 3000
    max_step= 200

    for i_episode in xrange(max_episode):
        print "{} spisode".format(i_episode+1)
        state = env.reset()
        for t in xrange(max_step):
            env.render()
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            print "state : ", state, " action : ", action, " next_state : ", next_state, " reward : ", reward, " done : ", done

            if done:
                print "Episode is ended. ({}time steps)".format(t+1)
                break
            else:
                state = next_state


if __name__ == "__main__":
    main()
