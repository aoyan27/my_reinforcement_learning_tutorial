#!/usr/bin/env python
#coding : utf-8

import numpy as np

import chainer 
from chainer import cuda, Variable, optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

class ValueIterationNetwork(chain):
    def __init__(self, n_in=2, l_h=150, l_q=8, n_out=8, k=10):
        super(ValueIterationNetwork, self).__init__(
            conv1 = L.Convolution2D(n_in, l_h, 3, stride=1, pad=1), 
            conv2 = L.Convolution2D(l_h, 1, 1, stride=1, pad=0, nobias=True),

            conv3a = L.Convolution2D(1, l_q, 3, stride=1, pad=1, nobias=True),
            conv3b = L.Convolution2D(1, l_q, 3, stride=1, pad=1, nobias=True),

            l4 = L.Linear(l_q, l_a, nobias=True),
        )

        self.k = k

    def __call__(self, x, state_x, state_y):
        pass

