#!/usr/bin/env python
#coding:utf-8

import numpy as np
np.set_printoptions(suppress=True, threshold=np.inf)

import copy
import sys

from envs.grid_world import Gridworld
from agents.a_star_agent import AstarAgent

def get_enemy_agent_action(env, agent_id=1):
    a_agent = AstarAgent(env)
    a_agent.get_shortest_path(state)
    if a_agent.found:
        #  print "a_agent.state_list : "
        #  print a_agent.state_list
        #  print "a_agent.shrotest_action_list : "
        #  print a_agent.shortest_action_list
        #  env.show_policy(a_agent.policy.transpose().reshape(-1))
        path_data = a_agent.show_path()
        print "view_path : "
        a_agent.view_path(path_data['vis_path'])

    

def main(rows, cols, noise, num_agent, seed):
    mode = 0
    env = Gridworld(rows, cols, num_agent, noise, seed=seed, mode=mode)

    print env.grid

    print "env.n_state : ", env.n_state
    print "env.n_action : ", env.n_action

    print "env._state : ", env._state
    
    env.set_start_random()
    env.set_goal_random()


    for i in xrange(1):
        print "================================="
        print "episode : ", i
        observation = env.reset()
        for j in xrange(10):
            print "-----------------------------"
            print "step : ", j
            print "state : ", observation
            env.show_objectworld_with_state()
            get_enemy_agent_action(env)
            action = env.get_sample_action()
            print "action : ", action
            observation, reward, episode_end, info = env.step(action)
            print "next_state : ", observation
            print "reward : ", reward
            print "episode_end : ", episode_end




if __name__ == "__main__":
    rows = 5
    cols = 5
    noise = 0.0
    num_agent = 2
    seed = 2

    main(rows, cols, noise, num_agent, seed)
