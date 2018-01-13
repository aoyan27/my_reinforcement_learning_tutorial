#!/usr/bin/env python
#coding:utf-8

import argparse

import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt
from progressbar import ProgressBar

import copy
import pickle

import tf

from envs.continuous_state_object_world import Objectworld
from agents.a_star_agent_for_continuous_object_world import AstarAgent
from agents.local_path_agent_for_continuous_object_world import LocalPlanAgent


def view_image(array, title):
    image = cv.cvtColor(array.astype(np.uint8), cv.COLOR_GRAY2RGB)
    #  print image
    plt.imshow(255 - 255*image, interpolation="nearest")
    plt.title(title)
    plt.show()

def save_dataset(data, filename):
    print "Save %d continuous_map_dataset.pkl!!!!!" % len(data['image'])
    with open(filename, mode='wb') as f:
        pickle.dump(data, f)

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

def get_reward_map(env):
    reward_map = np.zeros([env.rows, env.cols])
    discreate_goal = env.continuous2discreate(env.goal[0], env.goal[1])
    reward_map[discreate_goal] = env.R_max
    return reward_map

def euler2quaternion(yaw):
    q = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)
    #  print "q : ", q, type(q)
    return q

def get_trajs(env, n_trajs):
    domain_continuous_position_list = []
    domain_continuous_orientation_list = []
    domain_continuous_action_list = []

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
        #  ドメインの初期化(スタートと障害物を設置)
        traj_continuous_position_list = []
        traj_continuous_orientation_list = []
        traj_continuous_action_list = []

        env.clear_trajectory()
        env.set_start_random()
        #  print "env.start : ", env.start
        #  print "env.state_ : ", env.state_
        env.set_orientation_random(goal_heading=True)
        #  print "env.orientation_ : ", env.orientation_
        env.calc_continuous_trajectory()
        #  traj_continuous_orientation_list.append(env.orientation_)
        traj_continuous_orientation_list.append(euler2quaternion(env.orientation_))
        
        #  A*でdiscreateなglobal_pathを生成
        a_agent = AstarAgent(env)
        start_position = list(env.continuous2discreate(env.state_[0], env.state_[1]))
        #  print "start_position : ", start_position
        a_agent.get_shortest_path(start_position)
        path_data = None
        if a_agent.found:
            challenge_times = 0
            path_data = a_agent.show_path()
            #  print "view_path : "
            #  a_agent.view_path(path_data['vis_path'])
            
            #  global_pathに従ってlocal_pathを計算し、軌道、方位、行動を保存
            for i_step in xrange(200):
                #  print "=============================================="
                #  print "step : ", i_step

                l_agent = LocalPlanAgent(env, a_agent)
                l_agent.transform_global_path_discreate2continuous(path_data['state_list'])

                l_agent.get_future_trajectory(env.state_, env.orientation_)
                continuous_action \
                        = l_agent.evaluation_local_path(l_agent.future_traj_position_list, \
                        l_agent.future_traj_orientation_list, l_agent.continuous_global_path_list)
                #  print "l_agent.evaluation_value : ", l_agent.evaluation_value
                #  print "continuous_action : ", \
                        #  continuous_action, env.velocity_vector[continuous_action]
                traj_continuous_action_list.append(continuous_action)

                #  env.show_continuous_objectworld(\
                        #  global_path=l_agent.continuous_global_path_list, \
                        #  local_path=l_agent.future_traj_position_list, \
                        #  selected_path=l_agent.selected_traj_position_list)

                next_state, next_orientation, reward, done, info = env.step(continuous_action)
                #  traj_continuous_orientation_list.append(env.orientation_)
                traj_continuous_orientation_list.append(euler2quaternion(env.orientation_))
                #  print "next_state : ", next_state
                #  print "next_orientation : ", next_orientation
                #  print "reward : ", reward
                #  print "episode_end : ", done
                #  print info

                if done:
                    if reward > 0:
                        traj_continuous_action_list.append(continuous_action)
                        #  print "traj_continuous_action_list : ", len(traj_continuous_action_list)
                        traj_continuous_position_list \
                                = np.asarray([env.continuous_y_list, \
                                              env.continuous_x_list]).transpose(1, 0)
                        #  print "traj_continuous_position_list : ", \
                                #  len(traj_continuous_position_list)
                        #  print "traj_continuous_orientation_list : ", \
                                #  len(traj_continuous_orientation_list)
                        domain_continuous_position_list.append(traj_continuous_position_list)
                        domain_continuous_orientation_list.append(traj_continuous_orientation_list)
                        domain_continuous_action_list.append(traj_continuous_action_list)

                        j +=1
                    break

    if failed:
        del domain_continuous_position_list[:]
        del domain_continuous_orientation_list[:]
        del domain_continuous_action_list[:]

    #  print "domain_continuous_position_list : ", domain_continuous_position_list 
    #  print "domain_continuous_orientation_list : ", domain_continuous_orientation_list
    #  print "domain_continuous_action_list : ", domain_continuous_action_list

    return domain_continuous_position_list, \
           domain_continuous_orientation_list, \
           domain_continuous_action_list

def cvt_resize_image(image, cell_size, size):
    resize_image = np.zeros(size)
    index = np.asarray(np.where(image==1))
    continuous_index = index*cell_size
    resize_cell_size = float(image.shape[0]/size[0]) * cell_size
    resize_index = continuous_index / resize_cell_size
    discreate_resize_index = resize_index.astype(np.int8)
    resize_image[tuple(discreate_resize_index)] = 1

    return resize_image




def main(width, height, cell_size, resize_size, n_objects, n_domains, n_trajs, seed, dataset_path):
    rows = int(height / cell_size)
    cols = int(width / cell_size)
    goal = [height-1, width-1]
    print "Grid_size : ", (rows, cols)
    resize_size = (resize_size, resize_size)
    print "Resize_size : ", resize_size

    R_max = 1.0
    noise = 0.0

    env = Objectworld(rows, cols, cell_size, goal, R_max, noise, n_objects, seed, mode=1)

    max_samples = 200 * n_domains * n_trajs
    print "max_samples : ", max_samples
    image_data = np.zeros((max_samples, resize_size[0], resize_size[1]))
    reward_map_data = np.zeros((max_samples, resize_size[0], resize_size[1]))
    position_list_data = np.zeros((max_samples, 2))
    #  orientation_list_data = np.zeros(max_samples)
    orientation_list_data = np.zeros((max_samples, 4))
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
        env.set_goal_random()
        #  print "env.goal : ", env.goal
        #  env.set_objects(n_objects_random=False)
        env.set_objects()

        image = grid2image(env.grid)
        #  print "image : "
        #  print image.shape
        #  view_image(image, 'Gridworld')
        resize_image = cvt_resize_image(image, cell_size, resize_size)
        #  print "resize_image : "
        #  print resize_image.shape
        #  view_image(resize_image, 'Gridworld(resize)')
        reward_map = get_reward_map(env)
        #  print "reward_map : "
        #  print reward_map.shape
        #  view_image(reward_map, 'Reward map')
        resize_reward_map = cvt_resize_image(reward_map, cell_size, resize_size)
        #  print "resize_reward_map : "
        #  print resize_reward_map.shape
        #  view_image(resize_reward_map, 'Reward map(resize)')

        position_list, orientation_list, action_list = get_trajs(env, n_trajs)
        if len(position_list) == 0:
            continue

        ns = 0
        for i in xrange(n_trajs):
            #  print "num_sample : ", num_sample
            #  print "i : ", i
            #  print "len(position_list) : ", len(position_list)
            ns = len(position_list[i])
            #  print "ns : ", ns
            
            image_data[num_sample:num_sample+ns] = resize_image
            reward_map_data[num_sample:num_sample+ns] = resize_reward_map
            position_list_data[num_sample:num_sample+ns] = position_list[i][:]
            orientation_list_data[num_sample:num_sample+ns] = orientation_list[i][:]
            action_list_data[num_sample:num_sample+ns] = action_list[i][:]

            num_sample += ns

        prog.update(dom)
        dom += 1

    data = {}
    data['image'] = image_data[0:num_sample]
    data['reward'] = reward_map_data[0:num_sample]
    data['position'] = position_list_data[0:num_sample]
    data['orientation'] = orientation_list_data[0:num_sample]
    data['action'] = action_list_data[0:num_sample]
    
    print "Image size : ", data['image'][0].shape

    dataset_name = 'map_dataset_continuous.pkl'
    save_dataset(data, dataset_path+dataset_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script is make_map_dataset_continuous ...')
    
    parser.add_argument('-hei', '--height', default=10.0, type=float, \
            help='height of global gridworld(unit:[m])')
    parser.add_argument('-wid', '--width', default=10.0, type=float, \
            help='width of global gridworld(unit:[m])')
    parser.add_argument('-c', '--cell_size', default=0.10, type=float, \
            help='cell_size of gridworld(unit:[m])')
    parser.add_argument('-r', '--resize_size', default=20, type=int, \
            help='resize_size of grid_map')

    parser.add_argument('-o', '--n_objects', default=40, type=int, help='number of objects')
    parser.add_argument('-d', '--n_domains', default=5000, type=int, help='number of domains')
    parser.add_argument('-t', '--n_trajs', default=10, type=int, help='number of trajs')

    parser.add_argument('-s', '--seed', default=0, type=int, help='number of seed')

    parser.add_argument('-dp', '--dataset_path', default='datasets/', \
            type=str, help="save dataset directory")

    args = parser.parse_args()
    print args

    main(args.height, args.width, args.cell_size, args.resize_size, args.n_objects, \
            args.n_domains, args.n_trajs, args.seed, args.dataset_path)
