#!/usr/bin/env python
#coding:utf-8

import numpy as np
np.set_printoptions(suppress=True, threshold=np.inf)

import copy
import sys

from envs.multi_agent_grid_world import Gridworld
from agents.a_star_agent import AstarAgent

def get_enemy_agent_action(env, agent_id=1):
    a_agent = AstarAgent(env, agent_id)
    a_agent.get_shortest_path(env._state[agent_id], env.grid)
    #  a_agent.get_shortest_path(env._state[agent_id], env.agent_grid[agent_id])
    if a_agent.found:
        pass
        #  print "a_agent.state_list : "
        #  print a_agent.state_list
        #  print "a_agent.shrotest_action_list : "
        #  print a_agent.shortest_action_list
        #  env.show_policy(a_agent.policy.transpose().reshape(-1))
        path_data = a_agent.show_path()
        print "view_path_enemy : "
        a_agent.view_path(path_data['vis_path'])
    #  print "a_agent.shortest_action_list[0] : "
    #  print a_agent.shortest_action_list[0]
    action = int(a_agent.shortest_action_list[0])
    #  print "action : ", action

    return action

def get_my_agent_action(env, agent_id=0):
    a_agent = AstarAgent(env, agent_id)
    #  a_agent.get_shortest_path(env._state[agent_id], env.grid)
    a_agent.get_shortest_path(env._state[agent_id], env.agent_grid[agent_id])
    if a_agent.found:
        pass
        #  print "a_agent.state_list : "
        #  print a_agent.state_list
        #  print "a_agent.shrotest_action_list : "
        #  print a_agent.shortest_action_list
        #  env.show_policy(a_agent.policy.transpose().reshape(-1))
        path_data = a_agent.show_path()
        print "view_path_my : "
        a_agent.view_path(path_data['vis_path'])
    #  print "a_agent.shortest_action_list[0] : "
    #  print a_agent.shortest_action_list[0]
    action = int(a_agent.shortest_action_list[0])
    #  print "action : ", action

    return action
    

def main(rows, cols, noise, num_agent, seed):
    mode = 1
    env = Gridworld(rows, cols, num_agent, noise, seed=seed, mode=mode)

    print env.grid

    print "env.n_state : ", env.n_state
    print "env.n_action : ", env.n_action

    print "env._state : ", env._state
    
    #  env.set_start_random()
    #  env.set_goal_random()
    
    actions = {0: 0, 1: 0}

    for i in xrange(10):
        print "================================="
        print "episode : ", i

        observation = env.reset(random=True)
        #  observation = env.reset(random=False)
        for j in xrange(10):
            print "-----------------------------"
            print "step : ", j
            print "start : ", env.start
            print "goal : ", env.goal
            print "state : ", observation
            env.show_objectworld_with_state()
            print "env.agent_grid[0] : "
            print env.agent_grid[0]

            print "env.agent_grid[1] : "
            print env.agent_grid[1]

            actions[0] = my_action = get_my_agent_action(env)
            actions[1] = enemy_action = get_enemy_agent_action(env)
            #  actions[0] = my_action = env.get_sample_action_single_agent()
            print "actions : ", actions
            
            #  action = env.get_sample_action()
            #  print "action : ", action
            observation, reward, episode_end, info = env.step(actions)
            print "next_state : ", observation
            print "reward : ", reward
            print "episode_end : ", episode_end

            if (episode_end[0]==1 and episode_end[1]==1) or (episode_end[0]==2):
                break




if __name__ == "__main__":
    rows = 5
    cols = 5
    noise = 0.0
    num_agent = 2
    seed = 2

    main(rows, cols, noise, num_agent, seed)
