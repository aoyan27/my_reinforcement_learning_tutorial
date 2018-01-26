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
import math

import tf

#  from networks.vin_continuous_velocity_vector_attention_patch import ValueIterationNetworkAttention
from networks.vin_continuous_velocity_vector import ValueIterationNetwork
#  from networks.vin_continuous_velocity_vector_goal import ValueIterationNetwork
#  from networks.vin_continuous_velocity_vector_2 import ValueIterationNetwork
from envs.continuous_state_object_world import Objectworld
from agents.a_star_agent_for_continuous_object_world import AstarAgent
from agents.local_path_agent_for_continuous_object_world import LocalPlanAgent


def view_image(array, title):
    image = cv.cvtColor(array.astype(np.uint8), cv.COLOR_GRAY2RGB)
    #  print image
    plt.imshow(255 - 255*image, interpolation="nearest")
    plt.title(title)
    plt.show()

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

def load_model(model, filename):
    print "Load {}!!".format(filename)
    serializers.load_npz(filename, model)

def cvt_resize_image(image, cell_size, size):
    resize_image = np.zeros(size)
    index = np.asarray(np.where(image==1))
    continuous_index = index*cell_size
    resize_cell_size = float(image.shape[0])/float(size[0]) * cell_size
    resize_index = continuous_index / resize_cell_size
    discreate_resize_index = resize_index.astype(np.int8)
    resize_image[tuple(discreate_resize_index)] = 1

    return resize_image

def cvt_input_data(image, reward_map):
    input_data = \
            np.concatenate((np.expand_dims(image, 0), np.expand_dims(reward_map, 0)), axis=0)
    input_data = np.expand_dims(input_data, 0)
    #  print input_data.shape
    return input_data

def get_action(model, input_data, goal, position, orientation, velocity_vector):
    p = model(input_data, position, orientation, velocity_vector)
    #  p = model(input_data, goal, position, orientation, velocity_vector)
    print "p : ", p
    action = np.argmax(p.data)
    print "action : ", action
    return action

def calc_dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def main(width, height, cell_size, resize_size, n_objects, seed, gpu, model_path):
    rows = int(height / cell_size)
    cols = int(width / cell_size)
    goal = [height-1, width-1]
    print "Grid_size : ", (rows, cols)
    resize_size = (resize_size, resize_size)
    print "Resize_size : ", resize_size

    R_max = 1.0
    noise = 0.0

    env = Objectworld(rows, cols, cell_size, goal, R_max, noise, n_objects, seed, mode=1)

    #  model = ValueIterationNetworkAttention(l_q=9, n_out=11, k=25)

    model = ValueIterationNetwork(l_q=11, n_out=11, k=25)
    #  model = ValueIterationNetwork(l_q=9, n_out=11, k=20)
    load_model(model, model_path)
    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    i = 0

    #  goal_data = np.array([[0.0, 0.0]], dtype=np.float32)
    goal_data = np.array([[0.0]], dtype=np.float32)
    position_data = np.array([[0.0, 0.0]], dtype=np.float32)
    orientation_data = np.array([[0.0]], dtype=np.float32)
    orientation_quaternion_data = np.array([euler2quaternion(orientation_data)])
    velocity_vector_data = np.array([[0.0, 0.0]], dtype=np.float32)


    success_times = 0
    failed_times = 0
    while i < 5:
        print "i : ", i
        #  env.set_start_random()
        #  print "env.state_ : ", env.state_
        #  env.set_goal_random()
        #  env.set_objects()
        #  print "env.grid : "
        #  env.show_objectworld()
        position_data[0], orientation_data[0] = env.reset(random=True, goal_heading=True)
        orientation_quaternion_data[0] = euler2quaternion(orientation_data) 
        goal_data[0] = calc_dist(env.goal, env.state_)
        #  print "env.state_ : ", env.state_
        #  print "env.orientation_ : ", env.orientation_
        #  env.set_objects()
        print "goal_data : ", goal_data
        print "position_data : ", position_data
        print "orientation_data : ", orientation_data
        print "orientation_quaternion_data : ", orientation_quaternion_data
        env.calc_continuous_trajectory()

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

        input_data = cvt_input_data(resize_image, resize_reward_map)
        if gpu >= 0:
            input_data = cuda.to_gpu(input_data)
        print "input_data : ", input_data.shape


        for i_step in xrange(10000):
            print "=============================================="
            print "step : ", i_step
            print "position_data : ", position_data
            print "orientation_data : ", orientation_data
            print "velocity_vector_data : ", velocity_vector_data

            action = get_action(model, input_data, goal_data, \
                                position_data, orientation_quaternion_data, \
                                velocity_vector_data)

            #  continuous_action = env.get_action_sample(continuous=True)

            env.show_continuous_objectworld()

            next_state, orientation, reward, done, info = env.step(action)
            print "reward : ", reward
            print "episode_end : ", done
            print info
            goal_data[0] = calc_dist(env.goal, env.state_)
            position_data[0] = next_state
            orientation_data[0] = orientation
            orientation_quaternion_data[0] = euler2quaternion(orientation_data) 
            velocity_vector_data[0] = env.velocity_vector[action]

            if done:
                i += 1
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script is predict vin ...')
    parser.add_argument('-hei', '--height', default=10.0, type=float, \
            help='height of global gridworld(unit:[m])')
    parser.add_argument('-wid', '--width', default=10.0, type=float, \
            help='width of global gridworld(unit:[m])')
    parser.add_argument('-c', '--cell_size', default=0.25, type=float, \
            help='cell_size of gridworld(unit:[m])')
    parser.add_argument('-r', '--resize_size', default=20, type=int, \
            help='resize_size of grid_map')
    

    parser.add_argument('-o', '--n_objects', default=100, type=int, help='number of objects')
    parser.add_argument('-s', '--seed', default=0, type=int, help='number of random seed')
    parser.add_argument('-g', '--gpu', default=-1, type=int, help='number of gpu device')

    parser.add_argument('-m', '--model_path', default='models/vin_model_1.model', type=str, help="load model path")

    args = parser.parse_args()
    print args
    
    main(args.height, args.width, args.cell_size, args.resize_size, \
            args.n_objects, args.seed, args.gpu, args.model_path)



