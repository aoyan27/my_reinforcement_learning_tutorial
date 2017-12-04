#!/usr/bin/env python
#coding:utf-8

import argparse

import numpy as np
np.set_printoptions(suppress=True, threshold=np.inf)
import matplotlib.pyplot
import cv2 as cv
import matplotlib.pyplot as plt
from progressbar import ProgressBar

import chainer 
from chainer import cuda, Variable, optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

import copy
import pickle
import time

from networks.vin import ValueIterationNetwork
from envs.object_world import Objectworld
from envs.localgrid_objectworld import LocalgridObjectworld

from keyborad_controller import KeyboardController


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

def get_local_reward_map(local_grid, local_goal):
    reward_map = np.zeros(local_grid.shape)
    reward_map[tuple(local_goal)] = 1.0
    return reward_map

def cvt_input_data(image, reward_map):
    input_data = \
            np.concatenate((np.expand_dims(image, 0), np.expand_dims(reward_map, 0)), axis=0)
    #  print input_data.shape
    return input_data

def cvt_local_grid2input_data(local_grid, local_goal):
    image = grid2image(local_grid)
    #  print "image : "
    #  print image, image.shape
    reward_map = get_local_reward_map(local_grid, local_goal)
    #  print "reward_map : "
    #  print reward_map, reward_map.shape

    input_data = np.expand_dims(cvt_input_data(image, reward_map), 0)

    return input_data

def get_path(env, model, input_data, state_data, local_grid, local_goal):
    state_list = []
    action_list = []

    state_data_ = copy.deepcopy(state_data)
    #  print "state_data_ : ", state_data_
    state = state_data_[0]
    #  print "state : ", state
    #  print "local_goal : ", local_goal
    max_challenge_times = local_grid.shape[0] + local_grid.shape[1]
    challenge_times = 0
    resign = False
    while tuple(state) != local_goal:
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
        
        next_state, _, _ = env.ow.move(state, action, grid_range=[env.l_rows, env.l_cols])
        #  print "next_state : ", next_state

        state_list.append(list(state))
        action_list.append(action)

        state_data_[0] = next_state
        state = next_state
    state_list.append(list(state))

    #  print "state_list : ", state_list
    #  print "action_list : ", action_list 
    return state_list, action_list, resign

def show_path(env, state_list, action_list, local_grid, local_goal):
    n_local_state = local_grid.shape[0] * local_grid.shape[1]
    vis_path = np.array(['-']*n_local_state).reshape(local_grid.shape)
    index = np.where(local_grid == -1)
    vis_path[index] = '#'
    state_list = np.asarray(state_list)
    for i in xrange(len(state_list)):
        vis_path[tuple(state_list[i])] = '*'
    vis_path[int(local_grid.shape[0]/2), int(local_grid.shape[1]/2)] = '$'
    if len(state_list[state_list==local_goal]) == 2:
        vis_path[tuple(local_goal)] = 'G'

    path_data = {}
    path_data['vis_path'] = vis_path
    path_data['state_list'] = state_list
    path_data['action_list'] = action_list
    
    return path_data


def load_model(model, filename):
    print "Load {}!!".format(filename)
    serializers.load_npz(filename, model)

def main(rows, cols, n_objects, seed, l_rows, l_cols, gpu, model_path):
    g_goal = [rows-1, cols-1]
    R_max = 1.0
    noise = 0.0
    mode = 1

    l_goal_range = [l_rows, l_cols]

    env = LocalgridObjectworld(rows, cols, g_goal, R_max, noise, n_objects, seed, mode, \
            l_rows, l_cols, l_goal_range)
    #  print "env.global_world : "
    #  env.show_global_objectworld()
    reward_map = env.ow.grid.transpose().reshape(-1)
    
    
    model = ValueIterationNetwork(l_q=9, n_out=9, k=20)
    load_model(model, model_path)
    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    center_y = l_rows / 2
    #  print "center_y : ", center_y
    center_x = l_cols / 2
    #  print "center_x : ", center_x
    state_data = np.expand_dims(np.asarray([center_y, center_x]), 0)
    print "state_data : ", state_data 


    observation = env.reset()
    print observation[1]

    kc = KeyboardController(mode)

    env.show_global_objectworld()
    max_episode = 1
    max_step = 500
    for i_episode in xrange(max_episode):
        print "============================="
        print "episode : ", i_episode
        obsevation = env.reset()
        
        for i_step in xrange(max_step):
            print "---------------------------"
            print "step : ", i_step

            print "state : ", observation[0]
            
            env.show_global_objectworld()
            print "local_map : "
            print observation[1]
            print "local_goal : ", env.local_goal

            input_data = cvt_local_grid2input_data(observation[1], env.local_goal)
            #  print "input_data : "
            #  print input_data, input_data.shape
            if gpu >= 0:
                input_data = cuda.to_gpu(input_data)
            
            state_list, action_list, resign = \
                    get_path(env, model, input_data, state_data, observation[1], env.local_goal)
            path_data = show_path(env, state_list, action_list, observation[1], env.local_goal)
            print "path_data['vis_path'] : "
            print path_data['vis_path']

            #  action = env.get_sample_action()
            action = kc.controller()
            print "action : ", action, "(", env.ow.dirs[action], ")"

            observation, reward, done, info = env.step(action, reward_map)
            print "next_state : ", observation[0]

            print "reward : ", reward
            print "done : ", done, " (collision : ", env.ow.collisions_[action], ")"

            if done:
                break



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script is ... ')

    parser.add_argument('-r', '--rows', default=50, type=int, help='row of global gridworld')
    parser.add_argument('-c', '--cols', default=50, type=int, help='column of global gridworld')
    parser.add_argument('-o', '--n_objects', default=200, type=int, help='number of objects')
    parser.add_argument('-s', '--seed', default=3, type=int, help='seed')

    parser.add_argument('-l_r', '--l_rows', default=15, type=int, help='row of local gridworld')
    parser.add_argument('-l_c', '--l_cols', default=15, type=int, help='column of local gridworld')

    parser.add_argument('-g', '--gpu', default=-1, type=int, help='number of gpu device')
    parser.add_argument('-m', '--model_path', \
            default='models/vin_model_1.model', type=str, help='load model path')

    args = parser.parse_args()
    print args

    main(args.rows, args.cols, args.n_objects, args.seed, \
            args.l_rows, args.l_cols, args.gpu, args.model_path)
