#!/usr/bin/env python
#coding : utf-8

import numpy as np
np.set_printoptions(precision=1, suppress=True, threshold=np.inf)

import chainer 
from chainer import cuda, Variable, optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

class ValueIterationNetworkAttention(Chain):
    def __init__(self, n_in=2, l_h=150, l_q=5, n_out=5, k=10, net=None):
        super(ValueIterationNetworkAttention, self).__init__(
            conv1 = L.Convolution2D(n_in, l_h, 3, stride=1, pad=1, \
                initialW=net.conv1.W.data, initial_bias=net.conv1.b.data), 
            conv2 = L.Convolution2D(l_h, 1, 1, stride=1, pad=0, \
                initialW=net.conv2.W.data, nobias=True),

            conv3a = L.Convolution2D(1, l_q, 3, stride=1, pad=1, \
                    initialW=net.conv3a.W.data, nobias=True), 
            conv3b = L.Convolution2D(1, l_q, 3, stride=1, pad=1, \
                    initialW=net.conv3b.W.data, nobias=True), 

            l4 = L.Linear(None, 1024, nobias=True),
            l5 = L.Linear(1024, 258, nobias=True),
            l6 = L.Linear(258, n_out, nobias=True),
        )

        self.k = k
    
    def attention(self, v, state_list):
        #  print "v.data : ",
        #  print v.data[0]
        #  print "v.data.shape : ", v.data.shape
        patch_size = (3, 3)
        #  print "patch_size : ", patch_size
        w = np.zeros((v.data.shape[0], v.data.shape[1], patch_size[0], patch_size[1]))
        w_out = np.zeros(((v.data.shape[0], v.data.shape[1], patch_size[0]*patch_size[1])))
        #  print "w.shape : ", w.shape

        if isinstance(v.data, cuda.ndarray):
            w = cuda.to_gpu(w)

        center_y = int(patch_size[0] / 2)
        center_x = int(patch_size[1] / 2)
        #  print "(center_y, center_x) : ", center_y, center_x

        for i in xrange(v.data.shape[0]):
            #  print "========================================="
            #  print "i : ", i
            y, x = state_list[i]
            #  print "(y, x) : ", y, x
            min_y = int(y - center_y)
            max_y = int(y + center_y) + 1
            min_x = int(x - center_x)
            max_x = int(x + center_x) + 1
            #  print "min_y, max_y : ", min_y, max_y
            #  print "min_x, max_x : ", min_x, max_x
            if min_y < 0:
                min_y = 0
            if v.shape[2] < max_y:
                max_y =  v.shape[2]
            if min_x < 0:
                min_x = 0
            if v.shape[3] < max_x:
                max_x = v.shape[3]
            #  print "min_y, max_y : ", min_y, max_y
            #  print "min_x, max_x : ", min_x, max_x

            diff_min_y = min_y - y
            diff_max_y = max_y - y
            diff_min_x = min_x - x
            diff_max_x = max_x - x
            #  print "diff_min_y, diff_max_y : ", diff_min_y, diff_max_y
            #  print "diff_min_x, diff_max_x : ", diff_min_x, diff_max_x


            attention_min_y = int(center_y + diff_min_y)
            attention_max_y = int(center_y + diff_max_y)
            attention_min_x = int(center_x + diff_min_x)
            attention_max_x = int(center_x + diff_max_x)
            #  print "attention_min_y, attention_max_y : ", attention_min_y, attention_max_y
            #  print "attention_min_x, attention_max_x : ", attention_min_x, attention_max_x

            #  print "v.data[i] : "
            #  print v.data[i]
            #  print v.data[i, :, min_y:max_y, min_x:max_x]
            w[i, :, attention_min_y:attention_max_y, attention_min_x:attention_max_x] \
                    = v.data[i, :, min_y:max_y, min_x:max_x]
            #  print "w[i] : "
            #  print w[i]
            #  print F.flatten(w[i])
            #  print "w_out : ", w_out[i, :]
            w_out[i, :] = F.flatten(w[i]).data
        #  print "w_out : "
        #  print w_out
            
        #  print w.shape
        
        w_out = Variable(w_out.astype(np.float32))
        #  print "state_list : "
        #  print state_list[0]
        v =  w_out

        return v


    def __call__(self, input_data, state_list):
        input_data = Variable(input_data.astype(np.float32))

        h = F.relu(self.conv1(input_data))
        #  print "h : ", h
        self.r = self.conv2(h)
        #  print "self.r : ", self.r
        #  print "self.r : ", self.r.data.shape

        self.q = self.conv3a(self.r)
        #  print "q : ", q.data.shape
        
        self.v = F.max(self.q, axis=1, keepdims=True)
        #  print "self.v : ", self.v.shape

        for i in xrange(self.k):
            self.q = self.conv3a(self.r) + self.conv3b(self.v)
            self.v = F.max(self.q, axis=1, keepdims=True)

        #  print "q(after k) : ", q.shape
        #  print "q(after k) : ", q
        #  print "self.v : ", self.v

        v_out = self.attention(self.v, state_list)
        #  print "v_out : ", v_out

        h1 = self.l4(v_out)
        h2 = self.l5(h1)
        y = self.l6(h2)

        return y

    def forward(self, input_data, state_list, action_list):
        y = self.__call__(input_data, state_list)
        #  print "y : ", y
        
        t = Variable(action_list.astype(np.int32))

        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)



