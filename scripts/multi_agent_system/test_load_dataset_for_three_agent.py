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
import sys

import tf



def euler2quaternion(roll, pitch, yaw):
    q = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    return q

def quaternion2euler(q):
    e = tf.transformations.euler_from_quaternion(q)
    return e


def view_three_image(fig, array1, array2, array3, \
                     state1, state2, state3, \
                     action1, action2, action3, \
                     velocity1, velocity2, velocity3,\
                     orientation1, orientation2, orientation3,\
                     title='dataset viewer'):
    plt.clf()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    image1 = cv.cvtColor(array1.astype(np.uint8), cv.COLOR_GRAY2RGB)
    print "image1 : ", image1.shape
    image1[int(state1[0]), int(state1[1]), :] = [1, 0, 0]
    image2 = cv.cvtColor(array2.astype(np.uint8), cv.COLOR_GRAY2RGB)
    print "image2 : ", image2[int(state2[0]), int(state2[1])]
    image2[int(state2[0]), int(state2[1]), :] = [1, 0, 0]
    image3 = cv.cvtColor(array3.astype(np.uint8), cv.COLOR_GRAY2RGB)
    print "image3 : ", image3[int(state3[0]), int(state3[1])]
    image3[int(state3[0]), int(state3[1]), :] = [1, 0, 0]
    ax1.imshow(255 - 255*image1, interpolation="nearest")
    ax2.imshow(255 - 255*image2, interpolation="nearest")
    ax3.imshow(255 - 255*image3, interpolation="nearest")

    state_text1 = 'state : [%d, %d]' % (state1[0], state1[1])
    state_text2 = 'state : [%d, %d]' % (state2[0], state2[1])
    state_text3 = 'state : [%d, %d]' % (state3[0], state3[1])
    ax1.text(image1.shape[0]/4.0, image1.shape[1]+2, state_text1)
    ax2.text(image2.shape[0]/4.0, image2.shape[1]+2, state_text2)
    ax3.text(image3.shape[0]/4.0, image3.shape[1]+2, state_text3)

    action_text1 = 'action : %d' % (action1)
    action_text2 = 'action : %d' % (action2)
    action_text3 = 'action : %d' % (action3)
    ax1.text(image1.shape[0]/4.0, image1.shape[1]+3, action_text1)
    ax2.text(image2.shape[0]/4.0, image2.shape[1]+3, action_text2)
    ax3.text(image3.shape[0]/4.0, image3.shape[1]+3, action_text3)

    velocity_text1 = 'velocity : [%.3f, %.3f]' % (velocity1[0], velocity1[1])
    velocity_text2 = 'velocity : [%.3f, %.3f]' % (velocity2[0], velocity2[1])
    velocity_text3 = 'velocity : [%.3f, %.3f]' % (velocity3[0], velocity3[1])
    ax1.text(image1.shape[0]/4.0, image1.shape[1]+4, velocity_text1)
    ax2.text(image2.shape[0]/4.0, image2.shape[1]+4, velocity_text2)
    ax3.text(image3.shape[0]/4.0, image3.shape[1]+4, velocity_text3)

    orientation_text1 = 'orientation: %.3f' % (orientation1)
    orientation_text2 = 'orientation: %.3f' % (orientation2)
    orientation_text3 = 'orientation: %.3f' % (orientation3)
    ax1.text(image1.shape[0]/4.0, image1.shape[1]+5, orientation_text1)
    ax2.text(image2.shape[0]/4.0, image2.shape[1]+5, orientation_text2)
    ax3.text(image3.shape[0]/4.0, image3.shape[1]+5, orientation_text3)

    ax1.set_title(title+'1')
    ax2.set_title(title+'2')
    ax3.set_title(title+'3')
    plt.pause(1.0)

def get_diff_image(array1, array2):
    diff = array1 - array2
    index = np.where(diff == -1)
    diff[index] = 2
    print diff


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
    velocity_list_data = data['velocity']
    orientation_list_data = data['orientation']

    #  relative_orientation_list = data['relative_orientation']
    #  relative_velocity_vector_list = data['relative_velocity_vector']
    if len(grid_image_data) != 3:
        sys.stderr.write('\033[31m Error occurred!! No matching n_agents!! \033[0m \n')
        sys.exit(1)

    print "Load %d data!!!" % len(grid_image_data[0])

    #  return grid_image_data, agent_grid_image_data, another_agent_position_image_data, \
            #  reward_map_data, state_list_data, action_list_data

    return grid_image_data, agent_grid_image_data, another_agent_position_image_data, \
            reward_map_data, state_list_data, action_list_data, \
            velocity_list_data, orientation_list_data

    #  return grid_image_data, agent_grid_image_data, another_agent_position_image_data, \
            #  reward_map_data, state_list_data, orientation_list_data, action_list_data, \
            #  relative_orientation_list, relative_velocity_vector_list


def main(dataset, n_epoch, batchsize, gpu, model_path):
    #  grid_image_data, agent_grid_image_data, another_agent_position_image_data, \
            #  reward_map_data, state_list_data, orientation_list_data, action_list_data, \
            #  relative_orientation_list_data, relative_velocity_vector_list_data \
            #  = load_dataset(dataset)
    #  grid_image_data, agent_grid_image_data, another_agent_position_image_data, \
            #  reward_map_data, state_list_data, action_list_data \
            #  = load_dataset(dataset)

    grid_image_data, agent_grid_image_data, another_agent_position_image_data, \
            reward_map_data, state_list_data, action_list_data, \
            velocity_list_data, orientation_list_data\
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
        #  get_diff_image(agent_grid_image_data[0][i], agent_grid_image_data[1][i])
        #  view_two_image(fig, \
                       #  agent_grid_image_data[0][i], agent_grid_image_data[1][i],\
                       #  state_list_data[0][i], state_list_data[1][i], \
                       #  action_list_data[0][i], action_list_data[1][i])
        view_three_image(fig, \
                         agent_grid_image_data[0][i], \
                         agent_grid_image_data[1][i], \
                         agent_grid_image_data[2][i], \
                         state_list_data[0][i], \
                         state_list_data[1][i], \
                         state_list_data[2][i],\
                         action_list_data[0][i], \
                         action_list_data[1][i], \
                         action_list_data[2][i],\
                         velocity_list_data[0][i], \
                         velocity_list_data[1][i], \
                         velocity_list_data[2][i],\
                         orientation_list_data[0][i], \
                         orientation_list_data[1][i], \
                         orientation_list_data[2][i])
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

