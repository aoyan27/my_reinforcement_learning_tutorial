#!/usr/bin/env python
#coding:utf-8

import argparse

import gym, sys
import numpy as np

from agents.dac_agent import Agent

def main(env_name, gpu, render=False, monitor=True, evaluation=False, seed=0):
    env = gym.make(env_name)

    video_path = "/home/amsl/my_reinforcement_learning_tutorial/videos/dac_" + env_name
    model_path = "/home/amsl/my_reinforcement_learning_tutorial/models/deep_actor_critic/my_dac_" + env_name + "_"

    if monitor:
        env = gym.wrappers.Monitor(env, video_path, force=True)

    n_state = env.observation_space.shape[0]

    ### Pendulum-v0, MountainCarContinuous-v0 ###
    n_action = 1

    agent = Agent(n_state, n_action, gpu, seed)

    if evaluation:
        agent.load_model(model_path)
    
    max_episode = 10000
    max_step = 2000

    t = 0
    
    for i_episode in xrange(max_episode):
        observation = env.reset()
        r_sum = 0.0
        v_list = []
        a_list = []
        for j_step in xrange(max_step):
            if render:
                env.render()

            state = observation.astype(np.float32).reshape((1, n_state))
            #  print "state : ", state, type(state)

            #  action = env.action_space.sample()
            action, a = agent.get_action(state, evaluation)
            a_list.append(a)
            #  print "action : ", action, type(state)
            #  print "a : ", a

            observation, reward, episode_end, _ = env.step(action)
            next_state = observation.astype(np.float32).reshape((1, n_state))
            #  print "next_state : ", next_state, type(next_state)
            
            r_sum += reward
            
            if not evaluation:
                if t < agent.initial_exploration:
                    print "Initial exploration for critic(%d / %d)!!!!!!!!!!!" % (t, agent.initial_exploration)
                if agent.data_index_actor < agent.initial_exploration:
                    print "Initial exploration for actor(%d / %d)!!!!!!!!!!!" % (agent.data_index_actor, agent.initial_exploration)

            if not evaluation:
                agent.train(t, state, action, next_state, reward, episode_end)
                v_list.append(agent.V)

            t += 1

            if episode_end:
                break
        
        print "Episode : %d\t/Reward Sum : %f\t/Critic Loss : %f\t/Actor Loss : %f\t/Average V : %f\tTime Step : %d" % (i_episode+1, r_sum, agent.critic_loss, agent.actor_loss, sum(v_list)/float(t+1), agent.step)

        if not evaluation:
            agent.save_model(model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script is ...')
    parser.add_argument('--env', help='environment name', type=str, default='Pendulum-v0')
    parser.add_argument('--gpu', help='gpu mode(0) or cpu mode(-1)', type=int, default=-1)

    args = parser.parse_args()

    main(args.env, args.gpu, render=False)
    #  main(args.env, args.gpu, render=True, monitor=False, evaluation=True)
