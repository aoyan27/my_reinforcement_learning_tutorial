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

from envs.multi_agent_grid_world import Gridworld
from agents.dijkstra_agent import DijkstraAgent


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

def get_trajs(env, n_agents, n_trajs):
    domain_state_list = []
    domain_action_list = []
    
    failed = False

    j = 0
    challenge_times = 0
    while j < n_trajs:
        #  print "challenge_times : ", challenge_times
        challenge_times += 1
        if challenge_times > 50:
            failed = True
            break


    if failed:
        del domain_state_list[:]
        del domain_action_list[:]


    return domain_state_list, domain_action_list


def save_dataset(data, filename):
    print "Save %d map_dataset.pkl!!!!!" % len(data['image'])
    with open(filename, mode='wb') as f:
        pickle.dump(data, f)

def main(rows, cols, n_agents, n_domains, n_trajs, seed, save_dirs):
    n_state = rows * cols
    
    goal = [rows-1, cols-1]
    R_max = 1.0
    noise = 0.0
    mode = 1

    env = Gridworld(rows, cols, n_agents, noise, seed=seed, mode=mode)
    print env.grid

    #  print "env.n_state : ", env.n_state
    #  print "env.n_action : ", env.n_action

    print "env._state : ", env._state
    print "env.goal : ", env.goal

    max_samples = (rows + cols) * n_domains * n_trajs
    print "max_samples : ", max_samples

    image_data = np.zeros((max_samples, rows, cols))
    reward_map_data = np.zeros((n_agents, max_samples, rows, cols))
    state_list_data = np.zeros((n_agents, max_samples, 2))
    action_list_data = np.zeros((n_agents, max_samples))
    #  print "image_data : ", image_data.shape
    #  print "reward_map_data : ", reward_map_data.shape
    #  print "state_list_data : ", state_list_data.shape
    #  print "action_list_data : ", action_list_data.shape

    prog = ProgressBar(0, n_domains)

    dom = 0

    num_sample = 0
    while dom < n_domains:
        _ = env.reset(random=True)
        print "env._state : ", env._state
        print "env.goal : ", env.goal

        image = grid2image(env.grid)
        view_image(image, 'Gridworld')

        reward_map = get_reward_map(env, n_agents)

        state_list, action_list = get_trajs(env, n_agents, n_trajs)

        if len(state_list) == 0:
            continue

        prog.update(dom)
        dom += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script is make_dataset_multi_agent ...')
    
    parser.add_argument('-r', '--rows', default=9, type=int, help='row of global gridworld')
    parser.add_argument('-c', '--cols', default=9, type=int, help='column of global gridworld')

    parser.add_argument('-a', '--n_agents', default=2, type=int, help='number of agents')
    
    parser.add_argument('-d', '--n_domains', default=5000, type=int, help='number of domains')
    parser.add_argument('-t', '--n_trajs', default=10, type=int, help='number of trajs')
    

    parser.add_argument('-s', '--seed', default=0, type=int, help='number of seed')

    parser.add_argument('-m', '--dataset_dirs', default='datasets/', \
            type=str, help="save dataset directory")

    args = parser.parse_args()
    print args

    main(args.rows, args.cols, args.n_agents, args.n_domains, args.n_trajs, args.seed, args.dataset_dirs)


