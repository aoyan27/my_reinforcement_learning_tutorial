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

#  from networks.vin import ValueIterationNetwork
from networks.vin_with_orientation import ValueIterationNetwork
#  from envs.object_world import Objectworld
from envs.object_world_with_orientation import Objectworld

def euler_to_quaternion(roll, pitch, yaw):
    q = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    return q


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


def load_model(model, filename):
    print "Load {}!!".format(filename)
    serializers.load_npz(filename, model)

def create_input_data(image, reward_map):
    input_data = \
            np.concatenate((np.expand_dims(image, 0), np.expand_dims(reward_map, 0)), axis=0)
    #  print input_data.shape
    input_data = np.expand_dims(input_data, 0)
    #  print "input_data.shape : ", input_data.shape
    return input_data

def create_map_data(env, start, goal):
    env.set_objects()
    #  print "env.grid : "
    #  env.show_objectworld()
    image = grid2image(env.grid)
    #  print "image : "
    #  print image
    #  view_image(image, 'Gridworld')
    reward_map = get_reward_map(env)
    return image, reward_map

def get_reward_map(env):
    reward_map = np.zeros([env.rows, env.cols])
    reward_map[tuple(env.goal)] = env.R_max
    return reward_map


def main(rows, cols, n_objects, seed, gpu, model_path):
    model = ValueIterationNetwork(l_q=9, n_out=9, k=20)
    #  model = ValueIterationNetwork(l_h=200, l_q=9, n_out=9, k=20)
    load_model(model, model_path)
    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()
    
    n_state = rows * cols
    goal = [rows-1, cols-1]
    R_max = 1.0
    noise = 0.0

    #  env = Objectworld(rows, cols, goal, R_max, noise, n_objects, seed, mode=0)
    env = Objectworld(rows, cols, goal, R_max, noise, n_objects, seed, mode=1)
    
    # start, goal = set_start_and_goal(env)
    # print "start : ", start
    # print "goal : ", goal
    # image, reward_map = create_map_data(env, start, goal)
    # input_data = create_input_data(image, reward_map)
    #  print "input_data : "
    #  print input_data
    # print "input_data.shape : ", input_data.shape
    # state_data = np.expand_dims(np.asarray(start), 0)
    # print "state_data : ", state_data
    # print "state_data.shape : ", state_data.shape
    # if gpu >= 0:
    #     input_data = cuda.to_gpu(input_data)
    #
    # print "env.grid : "
    # env.show_objectworld()

    success_times = 0
    failed_times = 0
    
    max_episode = 100
    max_step = rows + cols
    prog = ProgressBar(0, max_episode)
    
    #  start_position = [0, 0]
    #  start_orientation = 0.0
     
    for i_episode in xrange(max_episode):
        prog.update(i_episode)
        print "=============================="
        print "episode : ", i_episode
        env.reset(orientation_list=[-180, -135, -90, -45, 0, 45, 90, 135, 180], random=True)
        position = env.state_
        print "position : ", position
        print "env.orientation_ : ", env.orientation_, math.degrees(env.orientation_)
        orientation = env.orientation_
        #  orientation = euler_to_quaternion(0.0, 0.0, env.orientation_)
        print "orientation : ", orientation
        goal = env.goal
        print "goal : ", env.goal

        state_data = np.expand_dims(np.asarray(position), 0)
        print "state_data : ", state_data
        orientation_data = np.expand_dims(np.asarray(orientation), 0)
        print "orientation_data : ", orientation_data

        image, reward_map = create_map_data(env, position, goal)
        #  print "image : "
        #  print image
        #  print "reward_map : "
        #  print reward_map
        input_data = create_input_data(image, reward_map)
        print "input_data : "
        print input_data

        
        print "env.grid : "
        env.show_objectworld()

        if gpu >= 0:
            input_data = cuda.to_gpu(input_data)
        for i_step in xrange(max_step):
            print "-----------------------------------"
            print "step : ", i_step
            print "state : ", state_data, ", goal : ", goal
            print "orientation_data : ", orientation_data
            env.show_objectworld_with_state()
            p = model(input_data, state_data, orientation_data)
            print "p : ", p
            action = np.argmax(p.data)
            #  action = env.get_action_sample()
            print "action : ",action
            next_state, reward, done, _ = env.step(action, reward_map.transpose().reshape(-1))
            #  print "next_state : ", next_state
            #  print "reward : ", reward
            #  print "done : ", done, " (collisions : ", env.collisions_[action], ")"
            print "env.orientation_ : ", env.orientation_

            state_data[0] = next_state
            orientation_data[0] = env.orientation_
            #  orientation_data[0] = euler_to_quaternion(0.0, 0.0, env.orientation_)

            if done:
                if reward == R_max:
                    success_times += 1
                break
        if reward != R_max:
            failed_times += 1
    
    print "success_times : ", success_times
    print "failed_times : ", failed_times




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script is predict vin ...')
    
    parser.add_argument('-r', '--rows', default=16, type=int, help='row of global gridworld')
    parser.add_argument('-c', '--cols', default=16, type=int, help='column of global gridworld')

    parser.add_argument('-o', '--n_objects', default=40, type=int, help='number of objects')
    parser.add_argument('-s', '--seed', default=0, type=int, help='number of random seed')
    parser.add_argument('-g', '--gpu', default=-1, type=int, help='number of gpu device')

    parser.add_argument('-m', '--model_path', default='models/vin_model_with_orientation_1.model', type=str, help="load model path")

    args = parser.parse_args()
    print args
    
    main(args.rows, args.cols, args.n_objects, args.seed, args.gpu, args.model_path)

