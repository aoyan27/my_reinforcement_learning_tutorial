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
import math

#  from envs.multi_agent_grid_world import Gridworld
from envs.multi_agent_object_world import Objectworld
from agents.a_star_agent import AstarAgent


def grid2image(array):
    image = copy.deepcopy(array)

    index = np.where(image == 1)
    for i in xrange(len(index[0])):
        image[index[0][i], index[1][i]] = 0

    index = np.where(image == -1)
    for i in xrange(len(index[0])):
        image[index[0][i], index[1][i]] = 1

    return image

def view_image(array, title):
    image = cv.cvtColor(array.astype(np.uint8), cv.COLOR_GRAY2RGB)
    #  print image
    plt.imshow(255 - 255*image, interpolation="nearest")
    plt.title(title)
    plt.show()

def get_input_image_and_traj(env, n_agents, n_trajs):
    stay_action = int(env.action_list[-1])

    domain_grid_list = {}
    domain_agent_grid_list = {}
    domain_another_agent_position_with_grid_list = {}
    domain_state_list = {}
    domain_action_list = {}
    for agent_id in xrange(n_agents):
        domain_grid_list[agent_id] = []
        domain_agent_grid_list[agent_id] = []
        domain_another_agent_position_with_grid_list[agent_id] = []
        domain_state_list[agent_id] = []
        domain_action_list[agent_id] = []

    actions = {}
    action = {}
    i_traj = 0
    #  print "env.grid : "
    #  env.show_array(env.grid)
    while i_traj < n_trajs:
        #  print "++++++++++++++++++++++ i_traj : ", i_traj, " ++++++++++++++++++++++++++++"
        traj_grid_list = {}
        traj_agent_grid_list = {}
        traj_another_agent_position_with_grid_list = {}
        traj_state_list = {}
        traj_action_list = {}
        for agent_id in xrange(n_agents):
            traj_grid_list[agent_id] = []
            traj_agent_grid_list[agent_id] = []
            traj_another_agent_position_with_grid_list[agent_id] = []
            traj_state_list[agent_id] = []
            traj_action_list[agent_id] = []

        max_step = 2*env.rows*env.cols
        #  print "max_step : ", max_step
        env.set_start_random(check_goal=True)
        observation = env.reset(start_position=env.start, goal_position=env.goal)
        #  print "observation : ", observation
        #  print "env.goal : ", env.goal
        
        #  initialize
        for agent_id in xrange(n_agents):
            traj_state_list[agent_id].append(observation[agent_id])
            traj_grid_list[agent_id].append(grid2image(env.grid))
            traj_agent_grid_list[agent_id].append(grid2image(env.agent_grid[agent_id]))
            another_agent_position_with_grid = np.zeros((env.rows, env.cols))
            for another_agent_id in xrange(n_agents):
                #  print "agent_id : ", agent_id
                if another_agent_id != agent_id:
                    #  print "another_agent_id : ", another_agent_id
                    another_agent_position = observation[another_agent_id]
                    #  print "another_agent_position : ", another_agent_position
                    another_agent_position_with_grid[tuple(another_agent_position)] = -1
            traj_another_agent_position_with_grid_list[agent_id].append(\
                    grid2image(another_agent_position_with_grid))
            #  print "traj_another_agent_position_with_grid_list : "
            #  print traj_another_agent_position_with_grid_list
        
        failed = True
        for i_step in xrange(max_step):
            #  print "--------------- i_step : ", i_step, " -----------------"
            for agent_id in xrange(n_agents):
                _, actions[agent_id], _ = get_agent_state_and_action(env, agent_id)
                action[agent_id] = int(actions[agent_id][0])
                traj_action_list[agent_id].append(action[agent_id])
            #  print "actions : ", actions
            #  print "action : ", action
            observation, reward, episode_end, info = env.step(action)
            #  print "observation : ", observation
            #  print "reward : ", reward
            #  print "episode_end : ", episode_end

            for agent_id in xrange(n_agents):
                traj_grid_list[agent_id].append(grid2image(env.grid))
                traj_agent_grid_list[agent_id].append(grid2image(env.agent_grid[agent_id]))
                another_agent_position_with_grid = np.zeros((env.rows, env.cols))
                for another_agent_id in xrange(n_agents):
                    #  print "agent_id : ", agent_id
                    if another_agent_id != agent_id:
                        #  print "another_agent_id : ", another_agent_id
                        another_agent_position = observation[another_agent_id]
                        #  print "another_agent_position : ", another_agent_position
                        another_agent_position_with_grid[tuple(another_agent_position)] = -1
                traj_another_agent_position_with_grid_list[agent_id].append(\
                        grid2image(another_agent_position_with_grid))
                #  print "traj_another_agent_position_with_grid_list : "
                #  print traj_another_agent_position_with_grid_list
                traj_state_list[agent_id].append(observation[agent_id])

            episode_end_flag_list = []
            for i in xrange(n_agents):
                episode_end_flag_list.append(episode_end[i] == 1)
            #  print "episode_end_flag_list : ", episode_end_flag_list
            if np.asarray(episode_end_flag_list).all():
                #  print "Success!!"
                for agent_id in xrange(n_agents):
                    traj_action_list[agent_id].append(stay_action)

                failed = False
                break

        if failed:
            #  print "failed!!"
            continue

        if not check_agent_collision(n_agents, traj_state_list):
            #  print "agent collision!!"
            continue

        i_traj += 1
            
        #  print "traj_grid_list[0] : ", np.asarray(traj_grid_list[0]).shape
        #  print "traj_state_list : ", np.asarray(traj_state_list[0]).shape
        #  print "traj_action_list : ", np.asarray(traj_action_list[0]).shape
        #  print "traj_grid_list[1] : ", np.asarray(traj_grid_list[1]).shape
        #  print "traj_state_list : ", np.asarray(traj_state_list[1]).shape
        #  print "traj_action_list : ", np.asarray(traj_action_list[1]).shape

        for agent_id in xrange(n_agents):
            domain_grid_list[agent_id].append(traj_grid_list[agent_id])
            domain_agent_grid_list[agent_id].append(traj_agent_grid_list[agent_id])
            domain_another_agent_position_with_grid_list[agent_id].append(\
                    traj_another_agent_position_with_grid_list[agent_id])
            domain_state_list[agent_id].append(traj_state_list[agent_id])
            domain_action_list[agent_id].append(traj_action_list[agent_id])
    #  print "domain_grid_list[0] : "
    #  print domain_grid_list[0]
    #  print len(domain_grid_list[0])
    #  print "domain_state_list[0] : "
    #  print domain_state_list[0]

    grid_list = []
    agent_grid_list = []
    another_agent_position_with_grid_list = []
    state_list = []
    action_list = []

    for agent_id in xrange(n_agents):
        grid_list.append(domain_grid_list[agent_id])
        agent_grid_list.append(domain_agent_grid_list[agent_id])
        another_agent_position_with_grid_list.append(\
                domain_another_agent_position_with_grid_list[agent_id])
        state_list.append(domain_state_list[agent_id])
        action_list.append(domain_action_list[agent_id])

    return grid_list, agent_grid_list, another_agent_position_with_grid_list, state_list, action_list


def get_reward_map(env, n_agents):
    reward_map = np.zeros((n_agents, env.rows, env.cols))
    for i in xrange(n_agents):
        reward_map[i, env.goal[i][0], env.goal[i][1]] = 1.0
    #  print "reward_map : "
    #  print reward_map
    return reward_map

def get_agent_state_and_action(env, agent_id):
    a_agent = AstarAgent(env, agent_id)
    #  print "env.agent_grid[agent_id] : "
    #  print env.agent_grid[agent_id]
    #  print "env._state[agent_id] : ", env._state[agent_id]
    #  a_agent.get_shortest_path(env._state[agent_id], env.grid)
    #  a_agent.get_shortest_path(env._state[agent_id], env.agent_grid[agent_id])
    a_agent.get_shortest_path(env._state[agent_id], env.agent_grid_future[agent_id])
    #  print "a_agent.found : ", a_agent.found
    if a_agent.found:
        #  pass
        #  print "a_agent.state_list : "
        #  print a_agent.state_list
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
    action_list = a_agent.shortest_action_list

    return state_list, action_list, a_agent.found

def check_agent_collision(n_agents, state_list):
    tmp_flag = None
    for agent_id in xrange(n_agents):
        agent_state_list = np.asarray(state_list[agent_id])
        for another_agent_id in xrange(agent_id+1, n_agents):
            another_agent_state_list = np.asarray(state_list[another_agent_id])
            #  print "agent_state_list : "
            #  print agent_state_list
            #  print "another_agent_state_list : "
            #  print another_agent_state_list

            tmp_flag = agent_state_list != another_agent_state_list
            #  print "tmp_flag : ", tmp_flag 

    agent_collision_flag = []
    no_ok_flag_pattern = [False, False]
    for flag in tmp_flag:
        agent_collision_flag.append(np.allclose(flag, no_ok_flag_pattern))
    #  print "agent_collision_flag : ", agent_collision_flag

    check_flag = None
    if np.asarray(agent_collision_flag).any():
        #  print "agent collision!!!!"
        check_flag = False
    else:
        #  print "agent not collision!!!!"
        check_flag =True

    return check_flag


def get_velocity_and_orientation(env, action_list):
    #  print "action_list : "
    #  print action_list

    n_agent = len(action_list)
    n_traj = len(action_list[0])
    velocity = [[] for i in xrange(n_agent)]
    orientation = [[] for i in xrange(n_agent)]
    for i in xrange(n_agent):
        traj_velocity = []
        for j in xrange(n_traj):
            velocity_ = [env.movement[a] for a in action_list[i][j]]
            end_velocity = velocity_.pop()
            velocity_.reverse()
            velocity_.append(end_velocity)
            velocity_.reverse()
            velocity[i].append(velocity_)
            orientation_ \
                = [math.atan2(v[0], v[1]) for v in velocity[i][j]]
            orientation_[0] = orientation_[1]
            orientation_size = len(orientation_)
            no_stay_index = np.where(np.asarray(action_list[i][j]) != 8)
            #  print "no_stay_index : ", no_stay_index[0][-1]
            #  print orientation_[no_stay_index[0][-1]]
            last_orientation = orientation_[no_stay_index[0][-1]]
            for k in xrange(no_stay_index[0][-1]+1, orientation_size):
                orientation_[k] = last_orientation
            orientation[i].append(orientation_)
            #  print "orientation_ : ", orientation_
    #  print "velocity : ", velocity
    #  print "orientation : ", orientation

    return velocity, orientation


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
    velocity_list_data = np.zeros((n_agents, max_samples, 2))
    orientation_list_data = np.zeros((n_agents, max_samples))
    
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
        #  env.set_objects(n_objects_random=False)
        #  print "env._state : ", env._state
        #  print "env.goal : ", env.goal
        #  print "env.grid : "
        #  print env.grid

        #  state_list, action_list = get_trajs(env, n_agents, n_trajs)
        grid_image_list, agent_grid_image_list, another_agent_position_with_grid_image_list, \
        state_list, action_list \
                = get_input_image_and_traj(env, n_agents, n_trajs)
        #  print "gri_image_list : "
        #  print grid_image_list
        #  view_image(image, 'Gridworld')

        if len(state_list) == 0:
            continue
        velocity_list, orientation_list = get_velocity_and_orientation(env, action_list)

        reward_map_list = get_reward_map(env, n_agents)
        #  print "reward_map_list : "
        #  print reward_map_list



        ns = 0
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
                velocity_list_data[j][num_sample[j]:num_sample[j]+ns] = velocity_list[j][i][:]
                orientation_list_data[j][num_sample[j]:num_sample[j]+ns] \
                        = orientation_list[j][i][:]

                num_sample[j] += ns

        prog.update(dom)
        dom += 1

    
    data = {'grid_image': [], 'agent_grid_image': [], 'another_agent_position': [], \
            'reward': [], 'state': [], 'action': [], 'velocity': [], 'orientation': []}
    for i in xrange(n_agents):
        data['grid_image'].append(grid_image_data[i][0:num_sample[i]])
        data['agent_grid_image'].append(agent_grid_image_data[i][0:num_sample[i]])
        data['another_agent_position'].append(\
                another_agent_position_with_grid_image_data[i][0:num_sample[i]])
        data['reward'].append(reward_map_data[i][0:num_sample[i]])
        data['state'].append(state_list_data[i][0:num_sample[i]])
        data['action'].append(action_list_data[i][0:num_sample[i]])
        data['velocity'].append(velocity_list_data[i][0:num_sample[i]])
        data['orientation'].append(orientation_list_data[i][0:num_sample[i]])
    
    dataset_name ='multi_agent_object_world_velocity_orientation_future_map_dataset.pkl'
    save_dataset(data, save_dirs+dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script is make_dataset_multi_agent ...')
    
    parser.add_argument('-r', '--rows', default=16, type=int, help='row of global gridworld')
    parser.add_argument('-c', '--cols', default=16, type=int, help='column of global gridworld')

    parser.add_argument('-o', '--n_objects', default=40, type=int, help='number of objects')
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
    
