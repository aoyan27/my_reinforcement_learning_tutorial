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
import pickle

import tf

#  from networks.vin import ValueIterationNetwork
from networks.vin_with_orientation import ValueIterationNetwork


def euler2quaternion(roll, pitch, yaw):
    q = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    return q

def quaternion2euler(q):
    e = tf.transformations.euler_from_quaternion(q)
    return e

def view_image(array, title):
    image = cv.cvtColor(array.astype(np.uint8), cv.COLOR_GRAY2RGB)
    #  print image
    plt.imshow(255 - 255*image, interpolation="nearest")
    plt.title(title)
    plt.show()


def load_dataset(path):
    data = None
    with open(path, mode='rb') as f:
        data = pickle.load(f)
    grid_image_data = data['grid_image']
    agent_grid_image_data = data['agent_grid_image']
    another_agent_position_image_data = data['another_agent_position']
    reward_map_data = data['reward']
    state_list_data = data['state']
    orientation_list_data = data['orientation']
    action_list_data = data['action']

    relative_orientation_list = data['relative_orientation']
    relative_velocity_vector_list = data['relative_velocity_vector']
    print "Load %d data!!!" % len(grid_image_data[0])

    return grid_image_data, agent_grid_image_data, another_agent_position_image_data, \
            reward_map_data, state_list_data, orientation_list_data, action_list_data, \
            relative_orientation_list, relative_velocity_vector_list

def train_test_split(grid_image_data, agent_grid_image_data, another_agent_position_image_data, \
        reward_map_data, state_list_data, orientation_list_data, action_list_data, \
        test_size=0.3, seed=0):

    np.random.seed(seed)
    n_dataset = grid_image_data[0].shape[0]
    #  print "n_dataset : ", n_dataset
    index = np.random.permutation(n_dataset)
    #  print "index : ", index

    n_test = int(test_size * n_dataset)
    #  print "n_test : ", n_test
    #  print "n_train : ", n_dataset - n_test
    index_test = index[0:n_test]
    index_train = index[n_test:]
    #  print "index_test : ", index_test
    #  print "index_train : ", index_train
    
    grid_image_test = grid_image_data[0][index_test]
    agent_grid_image_test = agent_grid_image_data[0][index_test]
    another_agent_position_image_test = another_agent_position_image_data[0][index_test]
    reward_map_test = reward_map_data[0][index_test]
    state_list_test = state_list_data[0][index_test]
    orientation_list_test = orientation_list_data[0][index_test]
    action_list_test = action_list_data[0][index_test]
    #  print "grid_image_test : ", len(grid_image_test)
    #  print "reward_map_test : ", len(reward_map_test)

    for i in xrange(1, len(grid_image_data)):
        grid_image_test = \
                np.concatenate([grid_image_test, grid_image_data[i][index_test]], axis=0)
        agent_grid_image_test = \
                np.concatenate([agent_grid_image_test, \
                agent_grid_image_data[i][index_test]], axis=0)
        another_agent_position_image_test = \
                np.concatenate([another_agent_position_image_test, \
                another_agent_position_image_data[i][index_test]], axis=0)
        reward_map_test = \
                np.concatenate([reward_map_test, reward_map_data[i][index_test]], axis=0)
        state_list_test = \
                np.concatenate([state_list_test, state_list_data[i][index_test]], axis=0)
        orientation_list_test = \
                np.concatenate([orientation_list_test, \
                orientation_list_data[i][index_test]], axis=0)
        action_list_test = \
                np.concatenate([action_list_test, action_list_data[i][index_test]], axis=0)
    #  print "grid_image_test : ", len(grid_image_test)
    #  print "reward_map_test : ", len(reward_map_test)

    grid_image_train = grid_image_data[0][index_train]
    agent_grid_image_train = agent_grid_image_data[0][index_train]
    another_agent_position_image_train = another_agent_position_image_data[0][index_train]
    reward_map_train = reward_map_data[0][index_train]
    state_list_train = state_list_data[0][index_train]
    orientation_list_train = orientation_list_data[0][index_train]
    action_list_train = action_list_data[0][index_train]
    #  print "grid_image_train : ", len(grid_image_train)
    #  print "reward_map_train : ", len(reward_map_train)

    for i in xrange(1, len(grid_image_data)):
        grid_image_train = \
                np.concatenate([grid_image_train, grid_image_data[i][index_train]], axis=0)
        agent_grid_image_train = \
                np.concatenate([agent_grid_image_train, \
                agent_grid_image_data[i][index_train]], axis=0)
        another_agent_position_image_train = \
                np.concatenate([another_agent_position_image_train, \
                another_agent_position_image_data[i][index_train]], axis=0)
        reward_map_train = \
                np.concatenate([reward_map_train, reward_map_data[i][index_train]], axis=0)
        state_list_train = \
                np.concatenate([state_list_train, state_list_data[i][index_train]], axis=0)
        orientation_list_train = \
                np.concatenate([orientation_list_train, \
                orientation_list_data[i][index_train]], axis=0)
        action_list_train = \
                np.concatenate([action_list_train, action_list_data[i][index_train]], axis=0)
    #  print "grid_image_train : ", len(grid_image_train)
    #  print "reward_map_train : ", len(reward_map_train)


    test_data = {}
    train_data = {}

    test_data['grid_image'] = grid_image_test
    test_data['agent_grid_image'] = agent_grid_image_test
    test_data['another_agent_position_image'] = another_agent_position_image_test
    test_data['reward'] = reward_map_test
    test_data['state'] = state_list_test 
    test_data['orientation'] = orientation_list_test 
    test_data['action'] = action_list_test

    train_data['grid_image'] = grid_image_train
    train_data['agent_grid_image'] = agent_grid_image_train
    train_data['another_agent_position_image'] = another_agent_position_image_train
    train_data['reward'] = reward_map_train
    train_data['state'] = state_list_train 
    train_data['orientation'] = orientation_list_train 
    train_data['action'] = action_list_train

    return train_data, test_data


def main(dataset, n_epoch, batchsize, gpu, model_path):
    grid_image_data, agent_grid_image_data, another_agent_position_image_data, \
            reward_map_data, state_list_data, orientation_list_data, action_list_data, \
            relative_orientation_list_data, relative_velocity_vector_list_data \
            = load_dataset(dataset)
    #  print "grid_image_data : ", len(grid_image_data[0])
    for i in xrange(len(grid_image_data[0])):
        print "============================================"
        print "state_list[1] : ", state_list_data[1][i]
        print "action_list[1] : ", action_list_data[1][i]
        print "orientation_list[1] : ", quaternion2euler(orientation_list_data[1][i])
        print "relative_orientatoin_list[1] : ", quaternion2euler(relative_orientation_list_data[1][i])
        print "state_list[0] : ", state_list_data[0][i]
        print "orientation_list[0] : ", quaternion2euler(orientation_list_data[0][i])
        print "relative_velocity_vector_list[0] : ", relative_velocity_vector_list_data[0][i]
        #  view_image(another_agent_position_image_data[0][i], 'map_image')
        view_image(agent_grid_image_data[0][i], 'map_image')
    #  print "orientation_list_data : ", len(orientation_list_data[0][0])
    #  print orientation_list_data
    print ""

    #  train_data, test_data = train_test_split(grid_image_data, agent_grid_image_data, \
            #  another_agent_position_image_data, reward_map_data, \
            #  state_list_data, orientation_list_data, action_list_data, test_size=0.3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script is training vin ...')

    parser.add_argument('-d', '--dataset', \
            default='datasets/multi_agent_object_world_with_orientation_map_dataset.pkl', \
            type=str, help="save dataset directory")

    parser.add_argument('-e', '--n_epoch', default=30, type=int, help='number of epoch')
    parser.add_argument('-b', '--batchsize', default=100, type=int, help='number of batchsize')
    parser.add_argument('-g', '--gpu', default=-1, type=int, help='number of gpu device')
    parser.add_argument('-m', '--model_path', \
            default='models/', type=str, help='model name')

    args = parser.parse_args()
    print args
    
    main(args.dataset, args.n_epoch, args.batchsize, args.gpu, args.model_path)

