#!/usr/bin/env python
#coding : utf-8

import numpy as np
np.set_printoptions(precision=1, suppress=True, threshold=np.inf)

import chainer 
from chainer import cuda, Variable, optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

class ValueIterationNetwork(Chain):
    def __init__(self, n_in=2, l_h=150, l_q=9, n_out=9, k=10):
        super(ValueIterationNetwork, self).__init__(
            conv1 = L.Convolution2D(n_in, l_h, 3, stride=1, pad=1), 
            conv2 = L.Convolution2D(l_h, 1, 1, stride=1, pad=0, nobias=True),

            conv3a = L.Convolution2D(1, l_q, 3, stride=1, pad=1, nobias=True),
            conv3b = L.Convolution2D(1, l_q, 3, stride=1, pad=1, nobias=True),

            l4 = L.Linear(None, 512, nobias=True),
            l5 = L.Linear(512, 258, nobias=True),
            l6 = L.Linear(258, n_out, nobias=True),
        )

        self.k = k
    
    def attention(self, q, position_list):
        #  print "q.data : ",
        #  print q.data[0]
        w = np.zeros(q.data.shape)
        #  print "w : ", w.shape
        cell_size = 0.5
        for i in xrange(len(position_list)):
            #  print "position_list : ", position_list[i]
            w[i, :, int(position_list[i][0]/cell_size), int(position_list[i][1]/cell_size)] = 1.0
            #  print "w : "
            #  print w[i]

        if isinstance(q.data, cuda.ndarray):
            w = cuda.to_gpu(w)

        #  print q.data.shape
        #  print w.shape
        
        w = Variable(w.astype(np.float32))
        #  print "position_list : "
        #  print position_list[0]
        a = q * w
        #  print "a : "
        #  print a.shape
        a = F.reshape(a, (a.data.shape[0], a.data.shape[1], -1))
        #  print "a() : "
        #  print a.shape

        q_out = F.sum(a, axis=2)
        #  print "q_out : "
        #  print q_out
        #  print q_out.shape
        return q_out


    def __call__(self, input_data, position_list, orientation_list, trigonometric_list, \
                 velocity_vector_list):
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

        #  print "q(after k) : ", q.shape
        #  print "q(after k) : ", q
        #  print "self.v : ", self.v
        
        q = self.conv3a(self.r) + self.conv3b(self.v)
        q_out = self.attention(q, position_list)

        #  print "q_out : ", q_out
        #  print "position_list : ", position_list
        #  print "orientation_list : ", orientation_list
        position_ = position_list.astype(np.float32)
        #  orientation_ = orientation_list.astype(np.float32)
        orientation_ = F.expand_dims(orientation_list.astype(np.float32), axis=1)
        input_policy = F.concat((position_, orientation_), axis=1)
        #  print "input_policy : ", input_policy
        trigonometric_ = trigonometric_list.astype(np.float32)
        input_policy2 = F.concat((input_policy, trigonometric_), axis=1)

        velocity_vector_ = velocity_vector_list.astype(np.float32)
        input_policy3 = F.concat((input_policy2, velocity_vector_), axis=1)
        #  input_policy2 = F.concat((position_, velocity_vector_), axis=1)

        h_in = F.concat((q_out, input_policy3), axis=1)
        #  print "h_in : ", h_in

        h1 = self.l4(h_in)
        h2 = self.l5(h1)
        y = self.l6(h2)

        return y

    def forward(self, input_data, position_list, orientation_list, trigonometric_list, \
                action_list, velocity_vector_list):
        y = self.__call__(input_data, position_list, orientation_list, trigonometric_list, \
                          velocity_vector_list)
        #  print "y : ", y
        
        t = Variable(action_list.astype(np.int32))

        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)



