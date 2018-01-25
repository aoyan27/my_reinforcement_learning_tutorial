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
    #  plt.show()
    plt.pause(0.05)

def view_two_image(fig, array1, array2, state1, state2, action1, action2, title):
    plt.clf()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    image2 = cv.cvtColor(array1.astype(np.uint8), cv.COLOR_GRAY2RGB)
    image1 = cv.cvtColor(array2.astype(np.uint8), cv.COLOR_GRAY2RGB)

    ax1.imshow(255 - 255*image1, interpolation="nearest")
    ax2.imshow(255 - 255*image2, interpolation="nearest")
    state_text1 = 'state : [%d, %d]' % (state1[0], state1[1])
    state_text2 = 'state : [%d, %d]' % (state2[0], state2[1])
    ax1.text(image1.shape[0]/4.0, image1.shape[1]+3, state_text1)
    ax2.text(image2.shape[0]/4.0, image2.shape[1]+3, state_text2)
    action_text1 = 'action : %d' % (action1)
    action_text2 = 'action : %d' % (action2)
    ax1.text(image1.shape[0]/4.0, image1.shape[1]+5, action_text1)
    ax2.text(image2.shape[0]/4.0, image2.shape[1]+5, action_text2)
    ax1.set_title(title+'1')
    ax2.set_title(title+'2')
    plt.pause(3.0)


def load_dataset(path):
    data = None
    with open(path, mode='rb') as f:
        data = pickle.load(f)
    grid_image_data = data['grid_image']
    agent_grid_image_data = data['agent_grid_image']
    another_agent_position_image_data = data['another_agent_position']
    reward_map_data = data['reward']
    state_list_data = data['state']
    #  orientation_list_data = data['orientation']
    action_list_data = data['action']

    #  relative_orientation_list = data['relative_orientation']
    #  relative_velocity_vector_list = data['relative_velocity_vector']
    print "Load %d data!!!" % len(grid_image_data[0])

    return grid_image_data, agent_grid_image_data, another_agent_position_image_data, \
            reward_map_data, state_list_data, action_list_data

    #  return grid_image_data, agent_grid_image_data, another_agent_position_image_data, \
            #  reward_map_data, state_list_data, orientation_list_data, action_list_data, \
            #  relative_orientation_list, relative_velocity_vector_list


def main(dataset, n_epoch, batchsize, gpu, model_path):
    #  grid_image_data, agent_grid_image_data, another_agent_position_image_data, \
            #  reward_map_data, state_list_data, orientation_list_data, action_list_data, \
            #  relative_orientation_list_data, relative_velocity_vector_list_data \
            #  = load_dataset(dataset)
    grid_image_data, agent_grid_image_data, another_agent_position_image_data, \
            reward_map_data, state_list_data, action_list_data \
            = load_dataset(dataset)
    #  print "grid_image_data : ", len(grid_image_data[0])
    fig = plt.figure()

    for i in xrange(len(grid_image_data[0])):
        print "============================================"
        print "state_list[1] : ", state_list_data[1][i]
        print "action_list[1] : ", action_list_data[1][i]
        #  print "orientation_list[1] : ", quaternion2euler(orientation_list_data[1][i])
        #  print "relative_orientatoin_list[1] : ", quaternion2euler(relative_orientation_list_data[1][i])
        print "state_list[0] : ", state_list_data[0][i]
        #  print "orientation_list[0] : ", quaternion2euler(orientation_list_data[0][i])
        #  print "relative_velocity_vector_list[0] : ", relative_velocity_vector_list_data[0][i]
        #  view_image(another_agent_position_image_data[0][i], 'map_image')
        #  view_image(agent_grid_image_data[0][i], 'map_image')
        #  view_image(agent_grid_image_data[1][i], 'map_image')
        view_two_image(fig, \
                       agent_grid_image_data[0][i], agent_grid_image_data[1][i],\
                       state_list_data[0][i], state_list_data[1][i], \
                       action_list_data[0][i], action_list_data[1][i], 'map_image')
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

