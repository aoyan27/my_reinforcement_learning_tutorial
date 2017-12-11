#!/usr/bin/env python
#coding:utf-8

import argparse

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from progressbar import ProgressBar

import copy
import pickle

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

def get_reward_map(env):
    reward_map = np.zeros([env.rows, env.cols])
    reward_map[tuple(env.goal)] = env.R_max
    return reward_map


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

    print "env.n_state : ", env.n_state
    print "env.n_action : ", env.n_action

    print "env._state : ", env._state


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


