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
from agents.dqn_agent import Agent

import time

def main(env_name, gpu, evaluation=False, monitor=True):
    env = gym.make(env_name)
    
    video_path = "/home/amsl/my_reinforcement_learning_tutorial/videos/dqn_" + env_name
    #  model_path = "/home/amsl/my_reinforcement_learning_tutorial/models/deep_q_learning/my_dqn_" + env_name + "/" + env_name + "_"
    #  model_path = "/subhdd/my_reinforcement_learning_tutorial/models/deep_q_learning/my_dqn_" + env_name + "/" + env_name + "_"
    model_path = "/media/amsl/HDCL-UT/my_reinforcement_learning_tutorial/models/deep_q_learning/my_dqn_" + env_name + "/" + env_name + "_"

    if monitor:
        env = gym.wrappers.Monitor(env, video_path, force=True)
    
    max_episode = 10001
    max_step = 2000

    num_state = env.observation_space.shape[0]
    #  print "num_state : ", num_state
    
    ####  Pendulum-v0  ####
    """
    action_list = [np.array([a]) for a in [-2.0, 2.0]]
    num_action = len(action_list)
    """

    ####  Acrobot-v1, CartPole-v0, MountainCar-v0  ####
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
        agent.load_model(model_path, 0)
    
    t = 0

    r_sum_list = []
    
    success = 0
    
    for i_episode in xrange(max_episode):
        observation = env.reset()
        q_list = []
        r_sum = 0.0
        for j_step in xrange(max_step):
            #  env.render()

            state = observation.astype(np.float32).reshape((1, num_state))
            #  print "state : ", state
            
            ####  Pendulum-v0  ####
            """
            act_i, q = agent.get_action(state, evaluation)
            action = action_list[act_i]
            """
            
            ####  Acrobot-v1, CartPole-v0, MountainCar-v0  ####
            action, q = agent.get_action(state, evaluation)
            #  print "action : ", action, type(action)


            q_list.append(q)

            observation, reward, done, _ = env.step(action)
            next_state = observation.astype(np.float32).reshape((1, num_state))
            #  print "next_state : ", next_state
            #  print "reward : ", reward
            #  print "exp_end : ", done
            
            if not evaluation:
                ####  Pendulum-v0  ####
                """
                agent.stock_experience(t, state, act_i, next_state, reward, done)
                """

                ####  Acrobot-v1, CartPole-v0, MountainCar-v0  ####
                agent.stock_experience(t, state, action, next_state, reward, done)

                agent.train(t)

            r_sum += reward

            t += 1

            if done:
                break
        print "Episode : %d\t Reward : %f\t Average Q : %f\t Loss : %f\t Epsilon : %f\t t : %d" % (i_episode+1, r_sum, sum(q_list)/float(t+1), agent.loss, agent.epsilon, t)

        if i_episode < 100:
            r_sum_list.append(r_sum)
        else:
            del r_sum_list[0]
            r_sum_list.append(r_sum)
            print "average 100 episode reward : ", sum(r_sum_list) / 100.0

        if r_sum == 200:
            success += 1
        print "Success : ", success, "\tSuccess rate : ", float(success)/float(i_episode+1)

        if not evaluation:
            agent.save_model(model_path, i_episode)
        else:
            agent.load_model(model_path, i_episode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script is ...')
    parser.add_argument('--env', help='environment name', type=str, default='Pendulum-v0')
    parser.add_argument('--gpu', help='gpu mode(0) or cpu mode(-1)', type=int, default=-1)

    args = parser.parse_args()

    #  main(args.env, args.gpu)
    main(args.env, args.gpu, evaluation=True, monitor=False)
