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

def main(env_name, gpu, evaluation=False, render=False, monitor=True):
    env = gym.make(env_name)
    
    video_path = "/home/amsl/my_reinforcement_learning_tutorial/videos/dqn_" + env_name
    model_path = "/home/amsl/my_reinforcement_learning_tutorial/models/deep_q_learning/" + env_name + "_"

    if monitor:
        env = gym.wrappers.Monitor(env, video_path, force=True)
    
    max_episode = 8001

    ####  Pendulum-v0  ####
    """
    max_step= 200
    """

    ####  Acrobot-v1, CartPole-v0  ####
    max_step = 500

    num_state = len(env.observation_space.high)
    
    ####  Pendulum-v0  ####
    """
    action_list = [np.array([a]) for a in [-2.0, 2.0]]
    num_action = len(action_list)
    """

    ####  Acrobot-v1, CartPole-v0  ####
    num_action = env.action_space.n

    agent = Agent(num_state, num_action, gpu)
    
    if gpu >= 0:
        print "GPU mode!!"
    else:
        print "CPU mode!!"

    if not evaluation:
        print "Train mode!!!"
    else:
        print "evaluation mode!!!"
        agent.load_model(model_path)
    
    t = 0

    for i_episode in xrange(max_episode):
        state = env.reset()
        q_list = []
        r_sum = 0.0
        for j_step in xrange(max_step):
            if render:
                env.render()

            state = np.array([state], dtype=np.float32)
            
            ####  Pendulum-v0  ####
            """
            act_i, q = agent.get_action(state, evaluation)
            action = action_list[act_i]
            """
            
            ####  Acrobot-v1, CartPole-v0  ####
            action, q = agent.get_action(state, evaluation)

            q_list.append(q)

            next_state, reward, done, _ = env.step(action)
            next_state = np.array([next_state], dtype=np.float32)
            
            if not evaluation:
                ####  Pendulum-v0  ####
                """
                agent.stock_experience(t, state, act_i, next_state, reward, done)
                """

                ####  Acrobot-v1, CartPole-v0  ####
                agent.stock_experience(t, state, action, next_state, reward, done)

                agent.train(t)

            r_sum += reward

            state = next_state

            t += 1

            if done:
                break
        print "Episode : %d\t Reward : %f\t Average Q : %f\t Loss : %f\t Epsilon : %f\t t : %d" % (i_episode+1, r_sum, sum(q_list)/float(t+1), agent.loss, agent.epsilon, t)
        
        if not evaluation:
            agent.save_model(model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script is ...')
    parser.add_argument('--env', help='environment name', type=str, default='Pendulum-v0')
    parser.add_argument('--gpu', help='gpu mode(0) or cpu mode(-1)', type=int, default=-1)

    args = parser.parse_args()

    main(args.env, args.gpu)
    #  main(args.env, args.gpu, evaluation=True, render=True, monitor=False)
