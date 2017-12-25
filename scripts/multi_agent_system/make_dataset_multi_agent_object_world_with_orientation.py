#!/usr/bin/env python
#coding:utf-8

import argparse

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from progressbar import ProgressBar

import copy
import pickle
import time

#  from envs.multi_agent_grid_world import Gridworld
#  from envs.multi_agent_object_world import Objectworld
from envs.multi_agent_object_world_with_orientation import Objectworld
#  from agents.a_star_agent import AstarAgent
from agents.a_star_agent_with_orientation import AstarAgent


def grid2image(array):
    #  print "array : "
    #  print array
    image = copy.deepcopy(array)

    index = np.where(image == 1)
    for i in xrange(len(index[0])):
        image[index[0][i], index[1][i]] = 0

    index = np.where(image == -1)
    for i in xrange(len(index[0])):
        image[index[0][i], index[1][i]] = 1
    #  print "image : "
    #  print image

    return image

def create_input_image(env, state_list, action_list, n_agents, n_trajs):
    #  print "======================================="
    #  print "env.grid : "
    #  print env.grid

    #  print "state_list : "
    #  print state_list
    #  print "len(state_list) : ", len(state_list[0][0])

    #  print "action_list : "
    #  print action_list

    #  print "n_agents : "
    #  print n_agents
    
    #  print "n_trajs : "
    #  print n_trajs


    domain_grid_list = {}
    domain_agent_grid_list = {}
    domain_another_agent_position_with_grid_list = {}
    for agent_id in xrange(n_agents):
        domain_grid_list[agent_id] = []
        domain_agent_grid_list[agent_id] = []
        domain_another_agent_position_with_grid_list[agent_id] = []

    start_position = {}
    goal_position = {}
    action = {}
    for i_traj in xrange(n_trajs):
        #  print "++++++++++++++++++++++++++++++++++++++++++++++++++"
        #  print "i_traj : ", i_traj
        traj_grid_list = {}
        traj_agent_grid_list = {}
        traj_another_agent_position_with_grid_list = {}
        for agent_id in xrange(n_agents):
            traj_grid_list[agent_id] = []
            traj_agent_grid_list[agent_id] = []
            traj_another_agent_position_with_grid_list[agent_id] = []


        max_step = len(state_list[0][i_traj])
        #  print "max_step : ", max_step
        for agent_id in xrange(n_agents):
            start_position[agent_id] = state_list[agent_id][i_traj][0]
            goal_position[agent_id] = state_list[agent_id][i_traj][max_step-1]
        #  print "start_position : ", start_position
        #  print "goal_position : ", goal_position
        observation = env.reset(start_position=start_position, goal_position=goal_position)
        #  print "env._state : ", env._state
        #  print "env.grid : "
        #  print env.grid
        #  for agent_id in xrange(n_agents):
            #  print "env.agent_grid[", agent_id, "] : "
            #  print env.agent_grid[agent_id]
        #  for agent_id in xrange(n_agents):
            #  print "env.another_agent_position_with_grid[", agent_id, "] : "
            #  print env.another_agent_position_with_grid[agent_id]
        for i_step in xrange(max_step):
            #  print "-----------------------------"
            #  print "step : ", i_step
            #  print "state : ", observation
            #  env.show_objectworld_with_state()
            #  print "env.grid : "
            #  print env.grid
            #  for agent_id in xrange(n_agents):
                #  print "env.agent_grid[", agent_id, "] : "
                #  print env.agent_grid[agent_id]
            #  for agent_id in xrange(n_agents):
                #  print "env.another_agent_position_with_grid[", agent_id, "] : "
                #  print env.another_agent_position_with_grid[agent_id]
            for agent_id in xrange(n_agents):
                traj_grid_list[agent_id].append(grid2image(env.grid))
                traj_agent_grid_list[agent_id].append(grid2image(env.agent_grid[agent_id]))
                traj_another_agent_position_with_grid_list[agent_id].append(\
                        grid2image(env.another_agent_position_with_grid[agent_id]))

            for agent_id in xrange(n_agents):
                action[agent_id] = int(action_list[agent_id][i_traj][i_step])
            #  print "action : ", action
            observation, reward, episode_end, info = env.step(action)
            #  print "next_state : ", observation
            #  print "reward : ", reward
            #  print "episode_end : ", episode_end
        #  print "traj_grid_list[0] : "
        #  print traj_grid_list[0]
        for agent_id in xrange(n_agents):
            domain_grid_list[agent_id].append(traj_grid_list[agent_id])
            domain_agent_grid_list[agent_id].append(traj_agent_grid_list[agent_id])
            domain_another_agent_position_with_grid_list[agent_id].append(\
                    traj_another_agent_position_with_grid_list[agent_id])

    #  print "domain_grid_list[0] : "
    #  print domain_grid_list[0]
    #  print len(domain_grid_list[0])

    grid_list = []
    agent_grid_list = []
    another_agent_position_with_grid_list = []

    for agent_id in xrange(n_agents):
        grid_list.append(domain_grid_list[agent_id])
        agent_grid_list.append(domain_agent_grid_list[agent_id])
        another_agent_position_with_grid_list.append(\
                domain_another_agent_position_with_grid_list[agent_id])

    #  print "grid_list : "
    #  print grid_list
    #  print len(grid_list)
    #  print "agent_grid_list : "
    #  print len(agent_grid_list)
    #  print "another_agent_position_with_grid_list : "
    #  print len(another_agent_position_with_grid_list[0][0])
    #  print another_agent_position_with_grid_list[0][0]
    
    return grid_list, agent_grid_list, another_agent_position_with_grid_list
    

def view_image(array, title):
    image = cv.cvtColor(array.astype(np.uint8), cv.COLOR_GRAY2RGB)
    #  print image
    plt.imshow(255 - 255*image, interpolation="nearest")
    plt.title(title)
    plt.show()

def get_reward_map(env, n_agents):
    reward_map = np.zeros((n_agents, env.rows, env.cols))
    for i in xrange(n_agents):
        reward_map[i, env.goal[i][0], env.goal[i][1]] = env.R_max
    #  print "reward_map : "
    #  print reward_map
    return reward_map

def get_agent_state_and_action(env, agent_id):
    print "************************************************"
    print "agent_id : ", agent_id
    a_agent = AstarAgent(env, agent_id)
    #  print "env.agent_grid[agent_id] : "
    #  print env.agent_grid[agent_id]
    #  print "env._state[agent_id] : ", env._state[agent_id]
    #  a_agent.get_shortest_path(env._state[agent_id], env.grid)
    a_agent.get_shortest_path(env._state[agent_id], env._orientation[agent_id], env.agent_grid[agent_id])
    #  print "a_agent.found : ", a_agent.found
    if a_agent.found:
        #  pass
        #  print "a_agent.state_list : "
        #  print a_agent.state_list
        #  print "a_agent.orientation_list : "
        #  print a_agent.orientation_list
        #  print "a_agent.shrotest_action_list : "
        #  print a_agent.shortest_action_list
        #  env.show_policy(a_agent.policy.transpose().reshape(-1))
        path_data = a_agent.show_path()
        #  print "agent_id : ", agent_id
        #  print "view_path_my : "
        #  a_agent.view_path(path_data['vis_path'])
    #  print "a_agent.shortest_action_list[0] : "
    #  print a_agent.shortest_action_list[0]
    state_list = a_agent.state_list
    orientation_list = a_agent.orientation_list
    action_list = a_agent.shortest_action_list
    print "state_list_ : ", state_list
    print "orientation_lsit_ : ", orientation_list
    print "action_list_ : ", action_list

    return state_list, orientation_list, action_list, a_agent.found

def map_all(es):
    return all([e == es[0] for e in es[1:]]) if es else False

def get_trajs(env, n_agents, n_trajs):
    state_list = []
    orientation_list = []
    action_list = []

    failed = False

    j = 0
    challenge_times = 0
    num_sample = [0 for i in xrange(n_agents)]
    found = [False for i in xrange(n_agents)]
    found_success = [True for i in xrange(n_agents)]
    while j < n_trajs:
        #  print "-----------------------------------------------"
        #  print "j : ", j
        #  print "challenge_times : ", challenge_times
        challenge_times += 1
        if challenge_times > 50:
            failed = True
            break
        env.set_start_random(check_goal=True)
        #  print "env.start_ : ", env.start
        #  print "env.goal_ : ", env.goal

        step_count_list = []
        for i in xrange(n_agents):
            domain_state_list, domain_orientation_list, domain_action_list, found[i] \
                    = get_agent_state_and_action(env, i)
            state_list.append(domain_state_list)
            action_list.append(domain_action_list)
            #  print "state_list : "
            #  print state_list
            #  print "action_list : "
            #  print action_list

            step_count_list.append(len(state_list[n_agents*j+i]))
        #  print "step_count_list : ", step_count_list
        if not map_all(step_count_list):
            max_index = np.argmax(np.asarray(step_count_list))
            max_value = np.max(np.asarray(step_count_list))
            #  print "max_value", max_value
            clone_count_list = []
            for i in xrange(n_agents):
                if i != max_index:
                    list_length = len(state_list[n_agents*j+i])
                    #  print "list_length : ", list_length
                    diff_length = max_value - len(state_list[n_agents*j+i])
                    #  print "diff_length : ", diff_length
                    for k in xrange(diff_length):
                        state_list[n_agents*j+i].append(state_list[n_agents*j+i][list_length-1])
                        action_list[n_agents*j+i].append(action_list[n_agents*j+i][list_length-1])
            #  print "clone_count_list : ", clone_count_list

        #  print "state_list_ : "
        #  print state_list
        #  print "action_list_ : "
        #  print action_list


        if found == found_success:
            j += 1
            challenge_times = 0

    if failed:
        del state_list[:]
        del action_list[:]
        return state_list, action_list
    
    agent_state_list = []
    agent_action_list = []
    for i in xrange(n_agents):
        agent_state_list.append(state_list[i::n_agents])
        agent_action_list.append(action_list[i::n_agents])
    #  print "agent_state_list : "
    #  print agent_state_list
    #  print "agent_action_list : "
    #  print agent_action_list
        
    return agent_state_list, agent_action_list


def save_dataset(data, filename):
    print "Save %d-%d multi_agent_map_dataset.pkl!!!!!" \
			% (len(data['grid_image'][0]), len(data['grid_image'][1]))
    with open(filename, mode='wb') as f:
        pickle.dump(data, f)


def main(rows, cols, n_objects, n_agents, n_domains, n_trajs, seed, save_dirs):
    n_state = rows * cols
    
    goal = [rows-1, cols-1]
    R_max = 1.0
    noise = 0.0
    mode = 1

    env = Objectworld(rows, cols, n_objects, n_agents, noise, seed=seed, mode=mode)
    #  print env.grid

    #  print "env.n_state : ", env.n_state
    #  print "env.n_action : ", env.n_action

    #  print "env._state : ", env._state
    #  print "env.goal : ", env.goal
    
    #  start = {0: [0, 0], 1: [rows-1, 0]}
    #  goal = {0: [rows-1, cols-1], 1:[0, cols-1]}
    #  env.set_start(start)
    #  env.set_goal(goal)
    
    max_samples = (rows + cols) * n_domains * n_trajs
    print "max_samples : ", max_samples

    grid_image_data = np.zeros((n_agents, max_samples, rows, cols))
    agent_grid_image_data = np.zeros((n_agents, max_samples, rows, cols))
    another_agent_position_with_grid_image_data = np.zeros((n_agents, max_samples, rows, cols))
    reward_map_data = np.zeros((n_agents, max_samples, rows, cols))
    state_list_data = np.zeros((n_agents, max_samples, 2))
    action_list_data = np.zeros((n_agents, max_samples))
    #  print "image_data : ", image_data.shape
    #  print "reward_map_data : ", reward_map_data.shape
    #  print "state_list_data : ", state_list_data.shape
    #  print "action_list_data : ", action_list_data.shape

    prog = ProgressBar(0, n_domains)

    dom = 0

    num_sample = [0 for i in xrange(n_agents)]
    while dom < n_domains:
        #  print "===================================================="
        #  print "dom : ", dom
        env.set_goal_random(check_start=False)
        env.set_objects()
        #  print "env._state : ", env._state
        #  print "env.goal : ", env.goal
        #  print "env.grid_ : "
        #  print env.grid


        reward_map_list = get_reward_map(env, n_agents)
        #  print "reward_map_list : "
        #  print reward_map_list

        state_list, action_list = get_trajs(env, n_agents, n_trajs)

        if len(state_list) == 0:
            continue

        grid_image_list, agent_grid_image_list, another_agent_position_with_grid_image_list \
                = create_input_image(env, state_list, action_list, n_agents, n_trajs)
        #  for k in xrange(len(another_agent_position_with_grid_image_list[0][0])):
            #  view_image(agent_grid_image_list[0][0][k], 'Gridworld')


        ns = 0
        count = 0
        for j in xrange(n_agents):
            #  print "num_sample[j] : ", num_sample[j]
            #  print "j : ", j
            #  print "len(state_list) : ", len(state_list)
            
            for i in xrange(n_trajs):
                #  print "i : ", i
                ns = len(state_list[j][i])
                #  print "ns : ", ns
                grid_image_data[j][num_sample[j]:num_sample[j]+ns] = grid_image_list[j][i][:]
                agent_grid_image_data[j][num_sample[j]:num_sample[j]+ns] \
                        = agent_grid_image_list[j][i][:]
                another_agent_position_with_grid_image_data[j][num_sample[j]:num_sample[j]+ns] \
                        = another_agent_position_with_grid_image_list[j][i][:]
                reward_map_data[j][num_sample[j]:num_sample[j]+ns] = reward_map_list[j]
                #  print "state_list : "
                #  print state_list[j][i][:]
                state_list_data[j][num_sample[j]:num_sample[j]+ns] = state_list[j][i][:]
                action_list_data[j][num_sample[j]:num_sample[j]+ns] = action_list[j][i][:]

                num_sample[j] += ns
                count += ns
        
        #  print max_samples
        #  print num_sample
        #  print "grid_image_data : "
        #  print grid_image_data[0][0:num_sample[0]]
        #  print "agent_grid_image_data : "
        #  print agent_grid_image_data[0][0:num_sample[0]]
        #  print "another_agent_position_with_grid_image_data : "
        #  print another_agent_position_with_grid_image_data[0][0:num_sample[0]]
        #  print "reward_map_data : "
        #  print reward_map_data[0][0:num_sample[0]]
        #  print state_list_data[0][0:num_sample[0]]
        #  print action_list_data[0]
        #  print max_samples
        #  print num_sample

        prog.update(dom)
        dom += 1

    
    data = {'grid_image': [], 'agent_grid_image': [], \
            'another_agent_position': [], 'reward': [], 'state': [], 'action': []}
    for i in xrange(n_agents):
        data['grid_image'].append(grid_image_data[i][0:num_sample[i]])
        data['agent_grid_image'].append(agent_grid_image_data[i][0:num_sample[i]])
        data['another_agent_position'].append(\
                another_agent_position_with_grid_image_data[i][0:num_sample[i]])
        data['reward'].append(reward_map_data[i][0:num_sample[i]])
        data['state'].append(state_list_data[i][0:num_sample[i]])
        data['action'].append(action_list_data[i][0:num_sample[i]])

    #  print len(data['grid_image'][1])
    #  print "data : "
    #  print data['grid_image']
    #  print data['agent_grid_image']
    #  print data['another_agent_position']
    #  print data['reward']
    #  print data['state']
    #  print data['action']
    
    dataset_name ='multi_agent_object_world_with_orientation_map_dataset.pkl'
    save_dataset(data, save_dirs+dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script is make_dataset_multi_agent ...')
    
    parser.add_argument('-r', '--rows', default=9, type=int, help='row of global gridworld')
    parser.add_argument('-c', '--cols', default=9, type=int, help='column of global gridworld')

    parser.add_argument('-o', '--n_objects', default=10, type=int, help='number of objects')
    parser.add_argument('-a', '--n_agents', default=2, type=int, help='number of agents')
    
    parser.add_argument('-d', '--n_domains', default=5000, type=int, help='number of domains')
    parser.add_argument('-t', '--n_trajs', default=10, type=int, help='number of trajs')
    

    parser.add_argument('-s', '--seed', default=0, type=int, help='number of seed')

    parser.add_argument('-m', '--dataset_dirs', default='datasets/', \
            type=str, help="save dataset directory")

    args = parser.parse_args()
    print args

    main(args.rows, args.cols, args.n_objects, args.n_agents, args.n_domains, \
            args.n_trajs, args.seed, args.dataset_dirs)
    
