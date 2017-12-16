#!/usr/bin/env python
#coding:utf-8

import argparse

import numpy as np
np.set_printoptions(suppress=True, threshold=np.inf)
import cv2 as cv
import matplotlib.pyplot as plt
from progressbar import ProgressBar

import chainer 
from chainer import cuda, Variable, optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

import copy
import sys
import pickle
import time

from networks.vin import ValueIterationNetwork
from envs.multi_agent_grid_world import Gridworld
from agents.a_star_agent import AstarAgent


def view_image(array, title):
    image = cv.cvtColor(array.astype(np.uint8), cv.COLOR_GRAY2RGB)
    #  print image
    plt.imshow(255 - 255*image, interpolation="nearest")
    plt.title(title)
    plt.show()

def get_enemy_agent_action(env, agent_id=1):
    a_agent = AstarAgent(env, agent_id)
    a_agent.get_shortest_path(env._state[agent_id], env.grid)
    #  a_agent.get_shortest_path(env._state[agent_id], env.agent_grid[agent_id])
    if a_agent.found:
        #  pass
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

def grid2image(array):
    image = copy.deepcopy(array)

    index = np.where(image == 1)
    for i in xrange(len(index[0])):
        image[index[0][i], index[1][i]] = 0

    index = np.where(image == -1)
    for i in xrange(len(index[0])):
        image[index[0][i], index[1][i]] = 1
    #  print image
    return image

def cvt_input_data(image, reward_map):
    input_data = \
            np.concatenate((np.expand_dims(image, 0), np.expand_dims(reward_map, 0)), axis=0)
    #  print input_data.shape
    input_data = np.expand_dims(input_data, 0)
    return input_data

def get_reward_map(env, agent_id=0):
    reward_map = np.zeros((env.rows, env.cols))
    reward_map[env.goal[agent_id][0], env.goal[agent_id][1]] = env.R_max
    #  print "reward_map : "
    #  print reward_map
    return reward_map

def get_my_agent_action(env, model, grid_map, reward_map, state, goal, gpu, agent_id=1):
    input_data = cvt_input_data(grid2image(grid_map), reward_map)
    #  print "input_data : "
    #  print input_data
    state_data = np.expand_dims(np.asarray(state), 0)

    if gpu >= 0:
        input_data = cuda.to_gpu(input_data)
        state_data = cuda.to_gpu(state_data)
    state_list, action_list, resign = \
            get_path(env, model, input_data, state_data, grid_map, goal)
    print "state_list : ", state_list
    print "resign : ", resign
    path_data = show_path(env, state_list, action_list, grid_map, goal)
    view_path(path_data['vis_path'])

    action = action_list[0]
    print "action : ", action
    return action

def get_path(env, model, input_data, state_data, grid, goal):
    state_list = []
    action_list = []

    state_data_ = copy.deepcopy(state_data)
    #  print "state_data_ : ", state_data_
    state = state_data_[0]
    #  print "state : ", state
    #  print "goal : ", goal
    max_challenge_times = grid.shape[0] + grid.shape[1]
    challenge_times = 0
    resign = False
    while tuple(state) != tuple(goal):
        challenge_times += 1
        if challenge_times >= max_challenge_times:
            #  state_list = []
            #  action_list = []
            resign = True
            break

        #  print "state_data_ : ", state_data_
        p = model(input_data, state_data_)
        #  print "p : ", p
        action = np.argmax(p.data)
        #  print "action : ", action, " (", env.ow.dirs[action], ")"
        
        next_state, _, _ = env.move(state, action)
        #  print "next_state : ", next_state

        state_list.append(list(state))
        action_list.append(action)

        state_data_[0] = next_state
        state = next_state
    state_list.append(list(state))

    #  print "state_list : ", state_list
    #  print "action_list : ", action_list 
    return state_list, action_list, resign

def show_path(env, state_list, action_list, grid, goal):
    n_local_state = grid.shape[0] * grid.shape[1]
    vis_path = np.array(['-']*n_local_state).reshape(grid.shape)
    index = np.where(grid == -1)
    vis_path[index] = '#'
    state_list = np.asarray(state_list)
    for i in xrange(len(state_list)):
        vis_path[tuple(state_list[i])] = '*'
    vis_path[tuple(state_list[0])] = '$'
    check_list = [1 if tuple(state)==tuple(goal) else 0 for state in state_list]
    if any(check_list):
        vis_path[tuple(goal)] = 'G'

    path_data = {}
    path_data['vis_path'] = vis_path
    path_data['state_list'] = state_list
    path_data['action_list'] = action_list
    
    return path_data

def view_path(path):
    grid = copy.deepcopy(path)
    for row in grid:
        print "|",
        for i in row:
            print "%2c" % i,
        print "|"


def load_model(model, filename):
    print "Load {}!!".format(filename)
    serializers.load_npz(filename, model)


def main(rows, cols, n_agents, seed, gpu, model_path):
    model = ValueIterationNetwork(l_q=9, n_out=9, k=20)
    load_model(model, model_path)
    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    noise = 0.0
    mode = 1
    env = Gridworld(rows, cols, n_agents, noise, seed=seed, mode=mode)

    print env.grid

    print "env.n_state : ", env.n_state
    print "env.n_action : ", env.n_action

    print "env._state : ", env._state
    
    #  env.set_start_random()
    #  env.set_goal_random()
    
    actions = {0: 0, 1: 0}
    """
    cross scenario
    """
    start = {0: [0, 0], 1: [8, 0]}
    goal = {0: [8, 8], 1:[0, 8]}

    """
    passing scenario
    """
    #  start = {0: [0, 0], 1: [8, 8]}
    #  goal = {0: [8, 8], 1:[0, 0]}
    
    """
    overtaking scenario
    """
    #  start = {0: [0, 0], 1: [1, 1]}
    #  goal = {0: [8, 8], 1:[5, 5]}


    success_times = 0
    failed_times = 0

    max_episode = 1
    max_step = rows + cols

    for i_episode in xrange(max_episode):
        print "================================="
        print "episode : ", i_episode

        #  observation = env.reset(random=True)
        observation = env.reset(start_position=start, goal_position=goal, random=False)
        reward_map = get_reward_map(env, agent_id=0)
        for i_step in xrange(max_step):
            print "-----------------------------"
            print "step : ", i_step
            print "start : ", env.start
            print "goal : ", env.goal
            print "state : ", observation
            env.show_objectworld_with_state()
            #  print "env.agent_grid[0] : "
            #  print env.agent_grid[0]

            #  print "env.agent_grid[1] : "
            #  print env.agent_grid[1]

            actions[0] = get_my_agent_action(env, model, env.agent_grid[0], reward_map, \
                    observation[0], env.goal[0], gpu)
            #  actions[0] = my_action = get_my_agent_action(env)
            #  actions[0] = env.get_sample_action_single_agent()
            actions[1] = get_enemy_agent_action(env)
            print "actions : ", actions
            
            #  actions = env.get_sample_action()
            #  print "action : ", actions
            observation, reward, episode_end, info = env.step(actions)
            print "next_state : ", observation
            print "reward : ", reward
            print "episode_end : ", episode_end
            time.sleep(0.5)

            if (episode_end[0]==1 and episode_end[1]==1) or (episode_end[0]==2):
                if episode_end[0]==1 and episode_end[1]==1:
                    success_times += 1
                break

        if episode_end[0]!=1 or episode_end[1]!=1:
            failed_times += 1

    print "success_times : ", success_times
    print "failed_times : ", failed_times
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script is predict vin ...')
    
    parser.add_argument('-r', '--rows', default=9, type=int, help='row of global gridworld')
    parser.add_argument('-c', '--cols', default=9, type=int, help='column of global gridworld')

    parser.add_argument('-a', '--n_agents', default=2, type=int, help='number of agents')
    parser.add_argument('-s', '--seed', default=0, type=int, help='number of random seed')
    parser.add_argument('-g', '--gpu', default=-1, type=int, help='number of gpu device')

    parser.add_argument('-m', '--model_path', default='models/multi_agent_vin_model_1.model', \
            type=str, help="load model path")

    args = parser.parse_args()
    print args

    main(args.rows, args.cols, args.n_agents, args.seed, args.gpu, args.model_path)
