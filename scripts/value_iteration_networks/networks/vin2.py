#!/usr/bin/env python
#coding : utf-8

import numpy as np
np.set_printoptions(precision=1, suppress=True, threshold=np.inf)
import cupy as xp

import chainer 
from chainer import cuda, Variable, optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

class ValueIterationNetwork(Chain):
    def __init__(self, n_in=2, l_h=150, l_q=5, n_out=5, k=10):
        super(ValueIterationNetwork, self).__init__(
            conv1 = L.Convolution2D(n_in, l_h, 3, stride=1, pad=1), 
            conv2 = L.Convolution2D(l_h, 1, 1, stride=1, pad=0, nobias=True),

            conv3a = L.Convolution2D(1, l_q, 3, stride=1, pad=1, nobias=True),
            conv3b = L.Convolution2D(1, l_q, 3, stride=1, pad=1, nobias=True),

            #  l4 = L.Linear(None, n_out, nobias=True),
             l4 = L.Linear(None, 1024, nobias=True),
             l5 = L.Linear(1024, 258, nobias=True),
             l6 = L.Linear(258, n_out, nobias=True),
        )

        self.k = k
    
    def attention(self, q, state_list):
        #  print "q : ", type(q), q.shape
        #  print "q.data : "
        #  print q.data[0]
        w = np.zeros(q.shape)
        extract_size = [1, 1]
        #  extract_size = [3, 3]
        attention_grid = \
            np.empty((q.shape[0], q.shape[1], extract_size[0], extract_size[1]))
        attention_grid.fill(1.0)

        if isinstance(q.data, cuda.ndarray):
            w = cuda.to_gpu(w)
            attention_grid = cuda.to_gpu(attention_grid)
        
        center_y = int(extract_size[0] / 2)
        center_x = int(extract_size[1] / 2)

        for i in xrange(q.data.shape[0]):
            #  print "========================================="
            #  print "i : ", i
            y, x = state_list[i]
            #  print "y, x : ", y, x
            min_y = int(y - center_y)
            max_y = int(y + center_y) + 1
            min_x = int(x - center_x)
            max_x = int(x + center_x) + 1
            if min_y < 0:
                min_y = 0
            if q.shape[2] < max_y:
                max_y =  q.shape[2]
            if min_x < 0:
                min_x = 0
            if q.shape[3] < max_x:
                max_x = q.shape[3]
            #  print "min_y, max_y : ", min_y, max_y
            #  print "min_x, max_x : ", min_x, max_x
            
            diff_min_y = min_y - y
            diff_max_y = max_y - y
            diff_min_x = min_x - x
            diff_max_x = max_x - x

            attention_min_y = int(center_y + diff_min_y)
            attention_max_y = int(center_y + diff_max_y)
            attention_min_x = int(center_x + diff_min_x)
            attention_max_x = int(center_x + diff_max_x)
            
            #  print "attentino_grid : "
            #  print attention_grid[i][:, attention_min_y:attention_max_y, attention_min_x:attention_max_x]

            w[i][:, min_y:max_y, min_x:max_x] = \
                    attention_grid[i]\
                    [:, attention_min_y:attention_max_y, attention_min_x:attention_max_x]
            #  print "w[i] : "
            #  print w[i]

        w = Variable(w.astype(np.float32))

        a = q * w
        #  print "a : "
        #  print a
        #  print a.shape
        a = F.reshape(a, (q.shape[0], q.shape[1], -1))
        #  print "a() : "
        #  print a
        #  print a.shape
        #  a = F.sum(a ,axis=2)
        #  a = Variable(attention_grid)
        #  print "a()() : "
        #  print a
        #  print a.shape
        
        return a


    def __call__(self, input_data, state_list):
        input_data = Variable(input_data.astype(np.float32))

        h = F.relu(self.conv1(input_data))
        #  print "h : ", h
        self.r = self.conv2(h)
        #  print "self.r : ", self.r
        #  print "self.r : ", self.r.data.shape

        q = self.conv3a(self.r)
        #  print "q : ", q.data.shape
        
        self.v = F.max(q, axis=1, keepdims=True)
        #  print "self.v : ", self.v.shape

        for i in xrange(self.k):
            q = self.conv3a(self.r) + self.conv3b(self.v)
            self.v = F.max(q, axis=1, keepdims=True)
        

        q = self.conv3a(self.r) + self.conv3b(self.v)
        q_out = self.attention(q, state_list)
        
        h1 = self.l4(q_out)
        h2 = self.l5(h1)
        y = self.l6(h2)

        #  y = self.l4(q_out)

        return y

    def forward(self, input_data, state_list, action_list):
        y = self.__call__(input_data, state_list)
        #  print "y : ", y
        
        t = Variable(action_list.astype(np.int32))

        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)



