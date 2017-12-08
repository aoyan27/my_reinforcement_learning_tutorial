#!/usr/bin/env python
#coding:utf-8

import argparse

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from progressbar import ProgressBar

import copy
import pickle

from envs.object_world import Objectworld
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

def get_trajs(env, n_trajs):
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

        #  print "---------------------------------------"
        #  print "j : ", j
        while 1:
            start_position = [np.random.randint(1, env.rows-1), np.random.randint(1, env.cols-1)]
            if start_position != env.goal and env.grid[tuple(start_position)] != -1:
                break
        #  print "start_position : ", start_position

        agent = DijkstraAgent(env)
        agent.get_shortest_path(start_position)
        
        if agent.found:
            #  print "agent.state_list : "
            #  print agent.state_list
            #  env.show_policy(agent.policy.transpose().reshape(-1))
            j += 1
            challenge_times = 0

            domain_state_list.append(agent.state_list)
            domain_action_list.append(agent.shortest_action_list)

        #  print "domain_state_list : "
        #  print domain_state_list
        #  print "domain_action_list : "
        #  print domain_action_list

    if failed:
        del domain_state_list[:]
        del domain_action_list[:]

    return domain_state_list, domain_action_list

def save_dataset(data, filename):
    print "Save %d map_dataset.pkl!!!!!" % len(data['image'])
    with open(filename, mode='wb') as f:
        pickle.dump(data, f)


def main(rows, cols, n_objects, n_domains, n_trajs, seed, save_dirs):
    n_state = rows * cols
    
    goal = [rows-1, cols-1]
    R_max = 1.0
    noise = 0.0

    #  env = Objectworld(rows, cols, goal, R_max, noise, n_objects, seed, mode=0)
    env = Objectworld(rows, cols, goal, R_max, noise, n_objects, seed, mode=1)
    
    max_samples = (rows + cols) * n_domains * n_trajs
    print "max_samples : ", max_samples
    image_data = np.zeros((max_samples, rows, cols))
    reward_map_data = np.zeros((max_samples, rows, cols))
    state_list_data = np.zeros((max_samples, 2))
    action_list_data = np.zeros(max_samples)
    #  print "image_data : ", image_data.shape
    #  print "reward_map_data : ", reward_map_data.shape
    #  print "state_list_data : ", state_list_data.shape
    #  print "action_list_data : ", action_list_data.shape

    prog = ProgressBar(0, n_domains)
   
    dom = 0

    num_sample = 0
    while dom < n_domains:
        #  print "=============================================="
        #  print "dom : ", dom
        goal = [np.random.randint(1, rows-1), np.random.randint(1, cols-1)]
        #  print "goal : ", goal
        env.set_goal(goal)

        env.set_objects()
        #  print "env.grid : "
        #  env.show_objectworld()
        image = grid2image(env.grid)
        #  print "image : "
        #  print image
        #  view_image(image, 'Gridworld')
        reward_map = get_reward_map(env)
        #  print "reward_map : "
        #  print reward_map

        state_list, action_list = get_trajs(env, n_trajs)
        if len(state_list) == 0:
            continue
        #  print "state_list : ", state_list[0]
        #  print "action_list : ", action_list[0]
            
        ns = 0
        for i in xrange(n_trajs):
            #  print "num_sample : ", num_sample
            #  print "i : ", i
            #  print "len(state_list) : ", len(state_list)
            ns = len(state_list[i])
            #  print "ns : ", ns
            
            image_data[num_sample:num_sample+ns] = image
            reward_map_data[num_sample:num_sample+ns] = reward_map
            state_list_data[num_sample:num_sample+ns] = state_list[i][:]
            action_list_data[num_sample:num_sample+ns] = action_list[i][:]

            num_sample += ns

        #  print image_data[0:num_sample]
        #  print reward_map_data[0:num_sample]
        #  print state_list_data[0:num_sample]
        #  print action_list_data[0:num_sample]
        #  print max_samples
        #  print num_sample
        
        prog.update(dom)
        dom += 1

    data = {}
    data['image'] = image_data[0:num_sample]
    data['reward'] = reward_map_data[0:num_sample]
    data['state'] = state_list_data[0:num_sample]
    data['action'] = action_list_data[0:num_sample]

    
    dataset_name ='map_dataset.pkl'
    save_dataset(data, save_dirs+dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script is make_map_dataset ...')
    
    parser.add_argument('-r', '--rows', default=16, type=int, help='row of global gridworld')
    parser.add_argument('-c', '--cols', default=16, type=int, help='column of global gridworld')

    parser.add_argument('-o', '--n_objects', default=40, type=int, help='number of objects')
    parser.add_argument('-d', '--n_domains', default=5000, type=int, help='number of domains')
    parser.add_argument('-t', '--n_trajs', default=10, type=int, help='number of trajs')

    parser.add_argument('-s', '--seed', default=0, type=int, help='number of seed')

    parser.add_argument('-m', '--dataset_dirs', default='datasets/', \
            type=str, help="save dataset directory")

    args = parser.parse_args()
    print args
    
    main(args.rows, args.cols, args.n_objects, args.n_domains, args.n_trajs, args.seed, args.dataset_dirs)
