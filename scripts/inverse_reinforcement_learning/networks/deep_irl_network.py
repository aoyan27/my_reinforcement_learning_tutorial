#!/usr/bin/env python
#coding:utf-8

import numpy as np
np.set_printoptions(suppress=True, threshold=np.inf)

import chainer 
from chainer import cuda, Variable, optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L


class DeepIRLNetwork(Chain):
    def __init__(self, n_in, n_out):
        super(DeepIRLNetwork, self).__init__(
                l1 = L.Linear(n_in, 1024),
                l2 = L.Linear(1024, 512),
                l3 = L.Linear(512, 256),
                #  l4 = L.Linear(256, n_out, initialW=np.zeros((n_out, 256), dtype=np.float32)),
                l4 = L.Linear(256, n_out),
                )

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        y = self.l4(h)
        return y

    def get_reward(self, feature):
        features = Variable(np.asarray(feature, dtype=np.float32))
        #  print "features.data : "
        #  print features.data
        reward = self.__call__(features)
        #  print "reward__ : "
        #  print reward
        #  reward = reward.data.reshape(-1)
        return reward
