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

from networks.vin import ValueIterationNetwork
from networks.vin_fine_tuning import ValueIterationNetworkFineTuning

class DelGradient(object):
    name = 'DelGradient'
    def __init__(self, delTgt):
        self.delTgt = delTgt

    def __call__(self, opt):
        for name,param in opt.target.namedparams():
            for d in self.delTgt:
                if d in name:
                    #  print "name : ", name
                    #  print "param : ", param
                    grad = param.grad
                    #  print "parame.grad : ", param.grad
                    with cuda.get_device(grad):
                        grad *= 0
                        #  print "grad : ", grad



def view_image(array, title):
    image = cv.cvtColor(array.astype(np.uint8), cv.COLOR_GRAY2RGB)
    #  print image
    plt.imshow(255 - 255*image, interpolation="nearest")
    plt.title(title)
    #  plt.show()
    plt.pause(0.05)


def load_model(model, filename):
    print "Load {}!!".format(filename)
    serializers.load_npz(filename, model)

def load_dataset(path):
    with open(path, mode='rb') as f:
        data = None
        data = pickle.load(f)
    image_data = data['image']
    reward_map_data = data['reward']
    state_list_data = data['state']
    action_list_data = data['action']
    print "Load %d data!!!" % len(image_data)

    return image_data, reward_map_data, state_list_data, action_list_data

def train_test_split(image_data, reward_map_data, state_list_data, action_list_data, \
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
    state_list_test = state_list_data[index_test]
    action_list_test = action_list_data[index_test]

    image_train = image_data[index_train]
    reward_map_train = reward_map_data[index_train]
    state_list_train = state_list_data[index_train]
    action_list_train = action_list_data[index_train]

    test_data = {}
    train_data = {}

    test_data['image'] = image_test
    test_data['reward'] = reward_map_test
    test_data['state'] = state_list_test 
    test_data['action'] = action_list_test

    train_data['image'] = image_train
    train_data['reward'] = reward_map_train  
    train_data['state'] = state_list_train 
    train_data['action'] = action_list_train

    return train_data, test_data

def cvt_input_data(image, reward_map):
    input_data = \
            np.concatenate((np.expand_dims(image, 1), np.expand_dims(reward_map, 1)), axis=1)
    #  print input_data.shape
    return input_data


def train_and_test(model, optimizer, gpu, model_path, train_data, test_data, n_epoch, batchsize, load_model=None):
    
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

            batch_state_list = train_data['state'][perm[i:i+batchsize \
                    if i+batchsize < n_train else n_train]]
            batch_action_list = train_data['action'][perm[i:i+batchsize \
                    if i+batchsize < n_train else n_train]]
            if gpu >= 0:
                batch_input_data = cuda.to_gpu(batch_input_data)
                batch_state_list = cuda.to_gpu(batch_state_list)
                batch_action_list = cuda.to_gpu(batch_action_list)

            real_batchsize = batch_image.shape[0]
            
            #  print "batch_input_data : ", batch_input_data[0]
            #  print "batch_state_list : ", batch_state_list[0]
            #  print "batch_action_list : ", batch_action_list[0]

            #  print "model.conv1.W.data : ", model.conv1.W.data[6]
            #  print "model.l4.W.data : ", model.l4.W.data[6]

            #  print "load_model.conv1.W.data : ", load_model.conv1.W.data[6]
            #  print "load_model.l4.W.data : ", load_model.l4.W.data[6]

            model.zerograds()
            loss, acc = model.forward(batch_input_data, batch_state_list, batch_action_list)
            #  print "loss(train) : ", loss
            loss.backward()
            optimizer.update()
            print "loss.grad ; ", loss.grad
            print "model.l4.W.grad : "
            print model.l4.W.grad
            print "model.conv3a.W.grad : "
            print model.conv3a.W.grad

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

            batch_state_list = test_data['state'][perm[i:i+batchsize \
                    if i+batchsize < n_test else n_test]]
            batch_action_list = test_data['action'][perm[i:i+batchsize \
                    if i+batchsize < n_test else n_test]]
            if gpu >= 0:
                batch_input_data = cuda.to_gpu(batch_input_data)
                batch_state_list = cuda.to_gpu(batch_state_list)
                batch_action_list = cuda.to_gpu(batch_action_list)

            real_batchsize = batch_image.shape[0]

            loss, acc = model.forward(batch_input_data, batch_state_list, batch_action_list)

            sum_test_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
            sum_test_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

            prog_test.update(i)

        print 'test mean loss={}, accuracy={}'\
                .format(sum_test_loss/n_test, sum_test_accuracy/n_test)         
        model_name = 'vin_model_%d.model' % epoch
        print model_name

        #  save_model(model, model_path+model_name)

        epoch += 1


def save_model(model, filename):
    print "Save {}!!".format(filename)
    serializers.save_npz(filename, model)


def main(dataset, n_epoch, batchsize, gpu, model_path, load_model_path):
    image_data, reward_map_data, state_list_data, action_list_data = load_dataset(dataset)
    #  print "image_data : ", image_data.shape
    #  for i in xrange(len(image_data)):
        #  print "i : ", i
        #  view_image(image_data[i], 'map_image')
        #  view_image(reward_map_data[i], 'reward_map')

    
    train_data, test_data = \
            train_test_split(image_data, reward_map_data, state_list_data, action_list_data, \
            test_size=0.3)
    
    load_model_ = ValueIterationNetwork(l_q=9, n_out=9, k=20)
    load_model(load_model_, load_model_path)

    #  model = ValueIterationNetwork(l_q=9, n_out=9, k=20)
    model = ValueIterationNetworkFineTuning(l_q=9, n_out=9, k=42, net=load_model_)
    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    optimizer = optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))
    optimizer.add_hook(chainer.optimizer.GradientClipping(100.0))
    optimizer.add_hook(DelGradient(["conv3a","conv3b"]))
    
    train_and_test(model, optimizer, gpu, model_path, train_data, test_data, n_epoch, batchsize, load_model=load_model_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script is training vin ...')

    parser.add_argument('-d', '--dataset', default='datasets/map_dataset.pkl', \
            type=str, help="save dataset directory")

    parser.add_argument('-e', '--n_epoch', default=30, type=int, help='number of epoch')
    parser.add_argument('-b', '--batchsize', default=100, type=int, help='number of batchsize')
    parser.add_argument('-g', '--gpu', default=-1, type=int, help='number of gpu device')
    parser.add_argument('-m', '--model_path', \
            default='models/', type=str, help='model name')
    parser.add_argument('-lm', '--load_model', \
            default='models/vin_model_1.model', type=str, help="load model path")

    args = parser.parse_args()
    print args
    
    main(args.dataset, args.n_epoch, args.batchsize, args.gpu, args.model_path, args.load_model)

