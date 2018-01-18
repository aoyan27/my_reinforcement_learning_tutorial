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

#  from networks.vin_continuous import ValueIterationNetwork
#  from networks.vin_continuous_velocity_vector import ValueIterationNetwork
#  from networks.vin_continuous_velocity_vector_2 import ValueIterationNetwork
from networks.vin_continuous_velocity_vector_2_attention import ValueIterationNetwork


velocity_vector \
        = {0: [0.5, -3.0], 1: [0.6, -2.5], 2: [0.7, -2.0], 3: [0.8, -1.5], 4: [1.0, -1.0], \
           5: [1.2, 0.0], \
           6: [1.0, 1.0], 7: [0.8, 1.5], 8: [0.7, 2.0], 9: [0.6, 2.5], 10: [0.5, 3.0]}

def view_image(array, title):
    image = cv.cvtColor(array.astype(np.uint8), cv.COLOR_GRAY2RGB)
    #  print image
    plt.imshow(255 - 255*image, interpolation="nearest")
    plt.title(title)
    plt.show()

def zscore(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore


def load_dataset(path):
    with open(path, mode='rb') as f:
        data = None
        data = pickle.load(f)

    image_data = data['image']
    reward_map_data = data['reward']
    position_list_data = data['position']
    orientation_list_data = data['orientation']
    action_list_data = data['action']
    velocity_vector_list_data_ = np.asarray([velocity_vector[i] for i in action_list_data])
    #  print "velocity_vector_list_data_ : ", velocity_vector_list_data_[0]
    tmp_velocity_vector_list_data = velocity_vector_list_data_[::-1]
    #  print "tmp_velocity_vector_list_data : ", tmp_velocity_vector_list_data[0]
    tmp_velocity_vector_list_data = np.append(tmp_velocity_vector_list_data, [[0.0, 0.0]], axis=0)
    #  print "tmp_velocity_vector_list_data : ", tmp_velocity_vector_list_data[0]

    velocity_vector_list_data = tmp_velocity_vector_list_data[::-1]
    velocity_vector_list_data_size = len(velocity_vector_list_data_)
    velocity_vector_list_data \
            = np.delete(velocity_vector_list_data, velocity_vector_list_data_size-1, axis=0)
    #  print "velocity_vector_list_data : ", velocity_vector_list_data[0]


    print "Load %d data!!!" % len(image_data)

    return image_data, reward_map_data, position_list_data, orientation_list_data, \
            action_list_data, velocity_vector_list_data

def save_model(model, filename):
    print "Save {}!!".format(filename)
    serializers.save_npz(filename, model)

def train_test_split(image_data, reward_map_data, \
                     position_list_data, orientation_list_data, \
                     action_list_data, velocity_vector_list_data, \
                     test_size, seed=0):
    np.random.seed(seed)
    n_dataset = image_data.shape[0]
    print "n_dataset : ", n_dataset
    index = np.random.permutation(n_dataset)
    #  print "index : ", index

    n_test = int(test_size * n_dataset)
    print "n_test : ", n_test
    print "n_train : ", n_dataset - n_test
    index_test = index[0:n_test]
    index_train = index[n_test:]
    #  print "index_test : ", index_test
    #  print "index_train : ", index_train

    image_test = image_data[index_test]
    reward_map_test = reward_map_data[index_test]
    position_list_test = position_list_data[index_test]
    orientation_list_test = orientation_list_data[index_test]
    action_list_test = action_list_data[index_test]
    velocity_vector_list_test = velocity_vector_list_data[index_test]

    image_train = image_data[index_train]
    reward_map_train = reward_map_data[index_train]
    position_list_train = position_list_data[index_train]
    orientation_list_train = orientation_list_data[index_train]
    action_list_train = action_list_data[index_train]
    velocity_vector_list_train = velocity_vector_list_data[index_train]

    test_data = {}
    train_data = {}

    test_data['image'] = image_test
    test_data['reward'] = reward_map_test
    test_data['position'] = position_list_test 
    test_data['orientation'] = orientation_list_test 
    test_data['action'] = action_list_test
    test_data['velocity_vector'] = velocity_vector_list_test

    train_data['image'] = image_train
    train_data['reward'] = reward_map_train  
    train_data['position'] = position_list_train 
    train_data['orientation'] = orientation_list_train 
    train_data['action'] = action_list_train
    train_data['velocity_vector'] = velocity_vector_list_train

    return train_data, test_data

def cvt_input_data(image, reward_map):
    input_data = \
            np.concatenate((np.expand_dims(image, 1), np.expand_dims(reward_map, 1)), axis=1)
    #  print input_data.shape
    return input_data

def train_and_test(model, optimizer, gpu, model_path, train_data, test_data, n_epoch, batchsize):
    epoch = 1
    accuracy = 0.0
    
    n_train = train_data['image'].shape[0]
    n_test = test_data['image'].shape[0]
    
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
            batch_image = train_data['image'][perm[i:i+batchsize \
                    if i+batchsize < n_train else n_train]]
            batch_reward_map = train_data['reward'][perm[i:i+batchsize \
                    if i+batchsize < n_train else n_train]]
            batch_input_data = cvt_input_data(batch_image, batch_reward_map)

            batch_position_list = train_data['position'][perm[i:i+batchsize \
                    if i+batchsize < n_train else n_train]]
            batch_orientation_list = train_data['orientation'][perm[i:i+batchsize \
                    if i+batchsize < n_train else n_train]]
            batch_action_list = train_data['action'][perm[i:i+batchsize \
                    if i+batchsize < n_train else n_train]]
            batch_velocity_vector_list = train_data['velocity_vector'][perm[i:i+batchsize \
                    if i+batchsize < n_train else n_train]]
            if gpu >= 0:
                batch_input_data = cuda.to_gpu(batch_input_data)
                batch_position_list = cuda.to_gpu(batch_position_list)
                batch_orientation_list = cuda.to_gpu(batch_orientation_list)
                batch_action_list = cuda.to_gpu(batch_action_list)
                batch_velocity_vector_list = cuda.to_gpu(batch_velocity_vector_list)
            #  print "batch_input_data : ", batch_input_data[0]
            #  print "batch_position_list : ", batch_position_list[0]
            #  print "batch_orientation_list : ", batch_orientation_list[0]
            #  print "batch_action_list : ", batch_action_list[0]

            real_batchsize = batch_image.shape[0]

            model.zerograds()
            loss, acc = model.forward(batch_input_data, \
                                      batch_position_list, batch_orientation_list, \
                                      batch_action_list, batch_velocity_vector_list)
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
            batch_image = test_data['image'][perm[i:i+batchsize \
                    if i+batchsize < n_test else n_test]]
            batch_reward_map = test_data['reward'][perm[i:i+batchsize \
                    if i+batchsize < n_test else n_test]]
            batch_input_data = cvt_input_data(batch_image, batch_reward_map)

            batch_position_list = test_data['position'][perm[i:i+batchsize \
                    if i+batchsize < n_test else n_test]]
            batch_orientation_list = train_data['orientation'][perm[i:i+batchsize \
                    if i+batchsize < n_test else n_test]]
            batch_action_list = test_data['action'][perm[i:i+batchsize \
                    if i+batchsize < n_test else n_test]]
            batch_velocity_vector_list = test_data['velocity_vector'][perm[i:i+batchsize \
                    if i+batchsize < n_test else n_test]]
            if gpu >= 0:
                batch_input_data = cuda.to_gpu(batch_input_data)
                batch_position_list = cuda.to_gpu(batch_position_list)
                batch_orientation_list = cuda.to_gpu(batch_orientation_list)
                batch_action_list = cuda.to_gpu(batch_action_list)
                batch_velocity_vector_list = cuda.to_gpu(batch_velocity_vector_list)

            real_batchsize = batch_image.shape[0]

            loss, acc = model.forward(batch_input_data, \
                                      batch_position_list, batch_orientation_list, \
                                      batch_action_list, batch_velocity_vector_list)

            sum_test_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
            sum_test_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

            prog_test.update(i)

        print 'test mean loss={}, accuracy={}'\
                .format(sum_test_loss/n_test, sum_test_accuracy/n_test)         
        model_name = 'vin_model_%d.model' % epoch
        print model_name

        save_model(model, model_path+model_name)

        epoch += 1


def main(dataset, n_epoch, batchsize, gpu, model_path):
    image_data, reward_map_data,  position_list_data, orientation_list_data, \
            action_list_data, velocity_vector_list = load_dataset(dataset)
    #  print "image_data[0] : ", image_data[0]
    #  print "reward_map_data[0] : ", reward_map_data[0]
    #  print "position_list_data[0] : ", position_list_data[0]
    #  print "orientation_list_data[0] : ", orientation_list_data[0]
    #  print "action_list_data[0] : ", action_list_data[0]
    #  print "image_data : ", image_data.shape
    #  view_image(image_data[0], 'map_image')
    
    #  print "reward_map_data : ", reward_map_data.shape
    #  view_image(reward_map_data[0], 'reward_map')
    
    train_data, test_data \
            = train_test_split(image_data, reward_map_data, \
                               position_list_data, orientation_list_data, \
                               action_list_data, velocity_vector_list, \
                               test_size=0.3)

    #  model = ValueIterationNetwork(l_q=5, n_out=5, k=20)
    model = ValueIterationNetwork(l_q=11, n_out=11, k=20)
    #  model = ValueIterationNetwork(l_h=200, l_q=9, n_out=9, k=20)
    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    optimizer = optimizers.Adam()
    optimizer.setup(model)
    #  optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))
    #  optimizer.add_hook(chainer.optimizer.GradientClipping(100.0))

    train_and_test(model, optimizer, gpu, model_path, train_data, test_data, n_epoch, batchsize)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script is training vin(continuous) ...')

    parser.add_argument('-d', '--dataset', default='datasets/map_dataset_continuous.pkl', \
            type=str, help="save dataset directory")

    parser.add_argument('-e', '--n_epoch', default=30, type=int, help='number of epoch')
    parser.add_argument('-b', '--batchsize', default=100, type=int, help='number of batchsize')
    parser.add_argument('-g', '--gpu', default=-1, type=int, help='number of gpu device')
    parser.add_argument('-m', '--model_path', \
                        default='models/', type=str, help='model name')

    args = parser.parse_args()
    print args
    
    main(args.dataset, args.n_epoch, args.batchsize, args.gpu, args.model_path)

