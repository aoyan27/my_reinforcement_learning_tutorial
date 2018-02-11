#!/usr/bin/env python
#coding : utf-8

import numpy as np
np.set_printoptions(precision=1, suppress=True, threshold=np.inf)

import chainer 
from chainer import cuda, Variable, optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

class ContinuousValueIterationNetwork(Chain):
    def __init__(self, n_in=2, l_h=150, l_q=9, n_out=9, k=10, net=None):
        super(ContinuousValueIterationNetwork, self).__init__(
            conv1 = L.Convolution2D(n_in, l_h, 3, stride=1, pad=1, \
                    initialW=net.conv1.W.data, initial_bias=net.conv1.b.data), 
            conv2 = L.Convolution2D(l_h, 1, 1, stride=1, pad=0, \
                    initialW=net.conv2.W.data, nobias=True),

            conv3a = L.Convolution2D(1, l_q, 3, stride=1, pad=1, \
                    initialW=net.conv3a.W.data, nobias=True),
            conv3b = L.Convolution2D(1, l_q, 3, stride=1, pad=1, \
                    initialW=net.conv3b.W.data, nobias=True),
            l4 = L.Linear(None, 9, initialW=net.l4.W.data, nobias=True),

            #  l5 = L.Linear(None, 1024, nobias=True),
            #  l6 = L.Linear(1024, 512, nobias=True),
            #  l7 = L.Linear(512, 256, nobias=True),
            #  l8 = L.Linear(256, n_out, nobias=True),

            l5 = L.Linear(None, 1024, nobias=True),
            l6 = L.Linear(1024, 512, nobias=True),
            l7 = L.Linear(512, 256, nobias=True),
            l8 = L.Linear(256, 128, nobias=True),
            l9 = L.Linear(128, 64, nobias=True),
            l10 = L.Linear(64, 32, nobias=True),
            l11 = L.Linear(32, n_out, nobias=True),
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


    def __call__(self, input_data, position_list, orientation_list, velocity_vector_list):
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
        
        discreate_out = self.l4(q_out)
        #  print "discreate_out : ", discreate_out

        #  print "q_out : ", q_out
        #  print "position_list : ", position_list
        #  print "orientation_list : ", orientation_list
        position_ = position_list.astype(np.float32)
        orientation_ = orientation_list.astype(np.float32)
        input_policy = F.concat((position_, orientation_), axis=1)
        #  print "input_policy : ", input_policy

        velocity_vector_ = velocity_vector_list.astype(np.float32)
        input_policy2 = F.concat((input_policy, velocity_vector_), axis=1)
        #  input_policy2 = F.concat((position_, velocity_vector_), axis=1)

        h_in = F.concat((discreate_out.data, input_policy2), axis=1)
        #  print "h_in : ", h_in

        #  h1 = F.leaky_relu(self.l5(h_in))
        #  h2 = F.leaky_relu(self.l6(h1))
        #  h3 = F.leaky_relu(self.l7(h2))
        #  y = self.l8(h3)

        h1 = F.leaky_relu(self.l5(h_in))
        h2 = F.leaky_relu(self.l6(h1))
        h3 = F.leaky_relu(self.l7(h2))
        h4 = F.leaky_relu(self.l8(h3))
        h5 = F.leaky_relu(self.l9(h4))
        h6 = F.leaky_relu(self.l10(h5))
        y = self.l11(h6)

        return y

    def forward(self, input_data, position_list, orientation_list, \
                action_list, velocity_vector_list):
        y = self.__call__(input_data, position_list, orientation_list, velocity_vector_list)
        #  print "y : ", y
        
        t = Variable(action_list.astype(np.int32))

        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)



