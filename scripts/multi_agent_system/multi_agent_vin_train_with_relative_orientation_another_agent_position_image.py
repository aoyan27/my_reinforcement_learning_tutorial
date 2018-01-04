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

#  from networks.vin import ValueIterationNetwork
from networks.vin_with_orientation import ValueIterationNetwork
#  from networks.vin_with_orientation_velocity import ValueIterationNetwork


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

    #  relative_orientation_list_data = data['relative_orientation']
    #  relative_velocity_vector_list_data = data['relative_velocity_vector']

    print "Load %d data!!!" % len(grid_image_data[0])

    return grid_image_data, agent_grid_image_data, another_agent_position_image_data, \
            reward_map_data, state_list_data, orientation_list_data, action_list_data, \
            #  relative_orientation_list_data, relative_velocity_vector_list_data


def train_test_split(grid_image_data, agent_grid_image_data, another_agent_position_image_data, \
        reward_map_data, state_list_data, orientation_list_data, action_list_data, \
        #  relative_orientation_list_data, relative_velocity_vector_list_data, \
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

    #  relative_orientation_list_test = relative_orientation_list_data[0][index_test]
    #  relative_velocity_vector_list_test = relative_velocity_vector_list_data[0][index_test]
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
        
        #  relative_orientation_list_test = \
                #  np.concatenate([relative_orientation_list_test, \
                #  relative_orientation_list_data[i][index_test]], axis=0)
        #  relative_velocity_vector_list_test = \
                #  np.concatenate([relative_velocity_vector_list_test, \
                #  relative_velocity_vector_list_data[i][index_test]], axis=0)

    #  print "grid_image_test : ", len(grid_image_test)
    #  print "reward_map_test : ", len(reward_map_test)

    grid_image_train = grid_image_data[0][index_train]
    agent_grid_image_train = agent_grid_image_data[0][index_train]
    another_agent_position_image_train = another_agent_position_image_data[0][index_train]
    reward_map_train = reward_map_data[0][index_train]
    state_list_train = state_list_data[0][index_train]
    orientation_list_train = orientation_list_data[0][index_train]
    action_list_train = action_list_data[0][index_train]

    #  relative_orientation_list_train = relative_orientation_list_data[0][index_train]
    #  relative_velocity_vector_list_train = relative_velocity_vector_list_data[0][index_train]
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

        #  relative_orientation_list_train = \
                #  np.concatenate([relative_orientation_list_train, \
                #  relative_orientation_list_data[i][index_train]], axis=0)
        #  relative_velocity_vector_list_train = \
                #  np.concatenate([relative_velocity_vector_list_train, \
                #  relative_velocity_vector_list_data[i][index_train]], axis=0)
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
    #  test_data['relative_orientation'] = relative_orientation_list_test
    #  test_data['relative_velocity_vector'] = relative_velocity_vector_list_test

    train_data['grid_image'] = grid_image_train
    train_data['agent_grid_image'] = agent_grid_image_train
    train_data['another_agent_position_image'] = another_agent_position_image_train
    train_data['reward'] = reward_map_train
    train_data['state'] = state_list_train 
    train_data['orientation'] = orientation_list_train 
    train_data['action'] = action_list_train
    #  train_data['relative_orientation'] = relative_orientation_list_train
    #  train_data['relative_velocity_vector'] = relative_velocity_vector_list_train

    return train_data, test_data


def cvt_input_data(grid_image, another_agent_position_image, reward_map):
    #  print "grid_image : "
    #  print grid_image[0]
    #  print "another_agent_position_image[0] : "
    #  print another_agent_position_image[0]
    #  print "reward_map[0] : "
    #  print reward_map[0]
    input_data = \
            np.concatenate((np.expand_dims(grid_image, 1), \
            np.expand_dims(another_agent_position_image, 1)), axis=1)

    #  print "input_data : "
    #  print input_data[0]

    input_data = \
            np.concatenate((input_data, np.expand_dims(reward_map, 1)), axis=1)

    #  print "input_data : "
    #  print input_data[0]

    return input_data


def train_and_test(model, optimizer, gpu, model_path, train_data, test_data, n_epoch, batchsize):
    epoch = 1
    accuracy = 0.0
    
    n_train = train_data['grid_image'].shape[0]
    n_test = test_data['grid_image'].shape[0]
    print "n_train : ", n_train
    print "n_test : ", n_test
    
    prog_train = ProgressBar(0, n_train)
    prog_test = ProgressBar(0, n_test)

    while epoch <= n_epoch:
        print "========================================="
        print "epoch : ", epoch
        sum_train_loss = 0.0
        sum_train_accuracy = 0.0

        perm = np.random.permutation(n_train)
        for i in xrange(0, n_train, batchsize):
            #  print " i : ", i
            batch_grid_image = train_data['grid_image'][perm[i:i+batchsize \
                    if i+batchsize < n_train else n_train]]
            batch_agent_grid_image = train_data['agent_grid_image'][perm[i:i+batchsize \
                    if i+batchsize < n_train else n_train]]
            batch_another_agent_position_image = \
                    train_data['another_agent_position_image'][perm[i:i+batchsize \
                    if i+batchsize < n_train else n_train]]
            batch_reward_map = train_data['reward'][perm[i:i+batchsize \
                    if i+batchsize < n_train else n_train]]
            #  batch_input_data = cvt_input_data(batch_grid_image, batch_reward_map)
            #  batch_input_data = cvt_input_data(batch_agent_grid_image, batch_reward_map)
            batch_input_data = cvt_input_data(batch_agent_grid_image, \
                    batch_another_agent_position_image, batch_reward_map)

            batch_state_list = train_data['state'][perm[i:i+batchsize \
                    if i+batchsize < n_train else n_train]]
            batch_orientation_list = train_data['orientation'][perm[i:i+batchsize \
                    if i+batchsize < n_train else n_train]]
            batch_action_list = train_data['action'][perm[i:i+batchsize \
                    if i+batchsize < n_train else n_train]]
            #  batch_relative_orientation_list = \
                    #  train_data['relative_orientation'][perm[i:i+batchsize \
                    #  if i+batchsize < n_train else n_train]]
            #  batch_relative_velocity_vector_list = \
                    #  train_data['relative_velocity_vector'][perm[i:i+batchsize \
                    #  if i+batchsize < n_train else n_train]]
            if gpu >= 0:
                batch_input_data = cuda.to_gpu(batch_input_data)
                batch_state_list = cuda.to_gpu(batch_state_list)
                batch_orientation_list = cuda.to_gpu(batch_orientation_list)
                batch_action_list = cuda.to_gpu(batch_action_list)
                #  batch_relative_orientation_list = cuda.to_gpu(batch_relative_orientation_list)
                #  batch_relative_velocity_vector_list = \
                        #  cuda.to_gpu(batch_relative_velocity_vector_list)

            real_batchsize = batch_grid_image.shape[0]
            
            #  print "batch_input_data : ", batch_input_data[0]
            #  print "batch_state_list : ", batch_state_list[0]
            #  print "batch_action_list : ", batch_action_list[0]

            model.zerograds()
            # loss, acc = model.forward(batch_input_data, batch_state_list, \
            #         batch_relative_orientation_list, batch_action_list)
            loss, acc = model.forward(batch_input_data, batch_state_list, \
                    batch_orientation_list, batch_action_list)

            #  print "loss(train) : ", loss
            loss.backward()
            optimizer.update()

            sum_train_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
            sum_train_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize
            
            prog_train.update(i)

        print 'train mean loss={}, accuracy={}'\
                .format(sum_train_loss/n_train, sum_train_accuracy/n_train)
        
        sum_test_loss = 0.0
        sum_test_accuracy = 0.0

        perm = np.random.permutation(n_test)
        for i in xrange(0, n_test, batchsize):
            batch_grid_image = test_data['grid_image'][perm[i:i+batchsize \
                    if i+batchsize < n_test else n_test]]
            batch_agent_grid_image = test_data['agent_grid_image'][perm[i:i+batchsize \
                    if i+batchsize < n_test else n_test]]
            batch_another_agent_position_image = \
                    test_data['another_agent_position_image'][perm[i:i+batchsize \
                    if i+batchsize < n_test else n_test]]
            batch_reward_map = test_data['reward'][perm[i:i+batchsize \
                    if i+batchsize < n_test else n_test]]
            #  batch_input_data = cvt_input_data(batch_grid_image, batch_reward_map)
            #  batch_input_data = cvt_input_data(batch_agent_grid_image, batch_reward_map)
            batch_input_data = cvt_input_data(batch_agent_grid_image, \
                    batch_another_agent_position_image, batch_reward_map)

            batch_state_list = test_data['state'][perm[i:i+batchsize \
                    if i+batchsize < n_test else n_test]]
            batch_orientation_list = test_data['orientation'][perm[i:i+batchsize \
                    if i+batchsize < n_test else n_test]]
            batch_action_list = test_data['action'][perm[i:i+batchsize \
                    if i+batchsize < n_test else n_test]]
            #  batch_relative_orientation_list = \
                    #  test_data['relative_orientation'][perm[i:i+batchsize \
                    #  if i+batchsize < n_test else n_test]]
            if gpu >= 0:
                batch_input_data = cuda.to_gpu(batch_input_data)
                batch_state_list = cuda.to_gpu(batch_state_list)
                batch_orientation_list = cuda.to_gpu(batch_orientation_list)
                batch_action_list = cuda.to_gpu(batch_action_list)
                #  batch_relative_orientation_list = cuda.to_gpu(batch_relative_orientation_list)

            real_batchsize = batch_grid_image.shape[0]

            # loss, acc = model.forward(batch_input_data, batch_state_list, \
            #         batch_relative_orientation_list, batch_action_list)
            loss, acc = model.forward(batch_input_data, batch_state_list, \
                    batch_orientation_list, batch_action_list)

            sum_test_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
            sum_test_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

            prog_test.update(i)

        print 'test mean loss={}, accuracy={}'\
                .format(sum_test_loss/n_test, sum_test_accuracy/n_test)         
        model_name = 'multi_agent_vin_model_%d.model' % epoch
        print model_name

        save_model(model, model_path+model_name)

        epoch += 1


def save_model(model, filename):
    print "Save {}!!".format(filename)
    serializers.save_npz(filename, model)


def main(dataset, n_epoch, batchsize, gpu, model_path):
    #  grid_image_data, agent_grid_image_data, another_agent_position_image_data, \
            #  reward_map_data, state_list_data, orientation_list_data, action_list_data, \
            #  relative_orientation_list_data, relative_velocity_vector_list_data \
            #  = load_dataset(dataset)
    grid_image_data, agent_grid_image_data, another_agent_position_image_data, \
            reward_map_data, state_list_data, orientation_list_data, action_list_data \
            = load_dataset(dataset)

    #  print "grid_image_data : ", len(grid_image_data[0])
    #  for i in xrange(len(grid_image_data[0])):
        #  view_image(another_agent_position_image_data[0][i], 'map_image')
    #  print "orientation_list_data : ", len(orientation_list_data[0][0])
    #  print orientation_list_data

    #  train_data, test_data = train_test_split(grid_image_data, agent_grid_image_data, \
            #  another_agent_position_image_data, reward_map_data, \
            #  state_list_data, orientation_list_data, action_list_data, \
            #  relative_orientation_list_data, relative_velocity_vector_list_data, \
            #  test_size=0.3)
    train_data, test_data = train_test_split(grid_image_data, agent_grid_image_data, \
            another_agent_position_image_data, reward_map_data, \
            state_list_data, orientation_list_data, action_list_data, \
            test_size=0.3)

    model = ValueIterationNetwork(n_in=3, l_q=9, n_out=9, k=20)
    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    optimizer = optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))
    optimizer.add_hook(chainer.optimizer.GradientClipping(100.0))


    train_and_test(model, optimizer, gpu, model_path, train_data, test_data, n_epoch, batchsize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script is training vin ...')

    parser.add_argument('-d', '--dataset', \
            default=\
            'datasets/multi_agent_object_world_with_orientation_velocity_map_dataset.pkl', \
            type=str, help="save dataset directory")

    parser.add_argument('-e', '--n_epoch', default=30, type=int, help='number of epoch')
    parser.add_argument('-b', '--batchsize', default=100, type=int, help='number of batchsize')
    parser.add_argument('-g', '--gpu', default=-1, type=int, help='number of gpu device')
    parser.add_argument('-m', '--model_path', \
            default='models/', type=str, help='model name')

    args = parser.parse_args()
    print args
    
    main(args.dataset, args.n_epoch, args.batchsize, args.gpu, args.model_path)

