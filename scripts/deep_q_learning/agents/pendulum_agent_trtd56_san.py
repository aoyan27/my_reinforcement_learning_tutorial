#!/usr/bin/env python
#coding:utf-8

import copy
import time

import numpy as np
import chainer
from chainer import cuda, optimizers, Variable, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L
from random import random, randint

import math

class Network(Chain):
    def __init__(self, n_in, n_out):
        super(Network, self).__init__(
                l1 = L.Linear(n_in, 100),
                l2 = L.Linear(100, 100),
                l3 = L.Linear(100, 100),
                l4 = L.Linear(100, n_out, initialW=np.zeros((n_out, 100), dtype=np.float32)),
                )

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        y = self.l4(h)
        return y

class Agent:
    ALPHA = 1e-6
    GAMMA = 0.99

    data_size = 1000
    replay_size = 100
    initial_exploration = 1000
    target_update_freq = 20

    def __init__(self, n_st, n_act, gpu, seed):
        np.random.seed(seed)
        self.n_st = n_st
        self.n_act = n_act

        self.gpu = gpu

        self.model = Network(self.n_st, self.n_act)
        self.target_model = copy.deepcopy(self.model)
        if self.gpu >= 0:
            self.model.to_gpu()
            self.target_model.to_gpu()
        
        self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, eps=0.0001)
        self.optimizer.setup(self.model)

        self.loss = 0
        self.step = 0

        self.epsilon = 1.0
        self.epsilon_decay = 0.00001
        self.epsilon_min = 0.1

        self.D = self.create_history_memory()
    
    def create_history_memory(self):
        st = np.zeros((self.data_size, 1, self.n_st), dtype=np.float32)
        act = np.zeros(self.data_size, dtype=np.float32)
        r = np.zeros(self.data_size, dtype=np.float32)
        st_dash = np.zeros((self.data_size, 1, self.n_st), dtype=np.float32)
        ep_end = np.zeros(self.data_size, dtype=np.bool)

        if self.gpu >= 0:
            st = cuda.to_gpu(st)
            act = cuda.to_gpu(act)
            r = cuda.to_gpu(r)
            st_dash = cuda.to_gpu(st_dash)
            ep_end = cuda.to_gpu(ep_end)
        return [st, act, r, st_dash, ep_end]

    def stock_experience(self, t, st, act, r, st_dash, ep_end):
        data_index = t % self.data_size
        if self.gpu >= 0:
            st = cuda.to_gpu(st)
            act = cuda.to_gpu(act)
            r = cuda.to_gpu(r)
            st_dash = cuda.to_gpu(st_dash)
            ep_end = cuda.to_gpu(ep_end)

        self.D[0][data_index] = st
        self.D[1][data_index] = act
        self.D[2][data_index] = r
        self.D[3][data_index] = st_dash
        self.D[4][data_index] = ep_end
    
    def forward(self, st, act, r, st_dash, ep_end):
        num_batch = self.replay_size

        s = Variable(st)
        s_dash = Variable(st_dash)

        Q = self.model(s)

        tmp = self.target_model(s_dash)

        max_Q_dash = map(np.max, tmp.data)
        #  print "max_Q_dash : ", max_Q_dash, type(max_Q_dash)
        target = copy.deepcopy(Q.data)

        for i in xrange(num_batch):
            if not ep_end[i]:
                tmp_ = r[i] + self.GAMMA*max_Q_dash[i]
            else:
                tmp_ = r[i]

            action_index = int(act[i][0])
            #  print "action_index : ", action_index

            target[i][action_index] = tmp_

        td = Variable(target) - Q
        
        zero_val = np.zeros((self.replay_size, self.n_act), dtype=np.float32)
        if self.gpu >= 0:
            zero_val = cuda.to_gpu(zero_val)
        zero_val = Variable(zero_val)

        loss = F.mean_squared_error(td, zero_val)
        self.loss = loss.data

        return loss, Q
    
    def experience_replay(self, t):
        if self.initial_exploration < t:
            if t < self.data_size:
                replay_index = np.random.randint(0, t, (self.replay_size, 1))
            else:
                replay_index = np.random.randint(0, self.data_size, (self.replay_size, 1))
            
            st_replay = np.ndarray(shape=(self.replay_size, 1, self.n_st), dtype=np.float32)
            act_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.float32)
            r_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.float32)
            st_dash_replay = np.ndarray(shape=(self.replay_size, 1, self.n_st), dtype=np.float32)
            ep_end_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.bool)
            if self.gpu >= 0:
                st_replay = cuda.to_gpu(st_replay)
                act_replay = cuda.to_gpu(act_replay)
                r_replay = cuda.to_gpu(r_replay)
                st_dash_replay = cuda.to_gpu(st_dash_replay)
                ep_end_replay = cuda.to_gpu(ep_end_replay)

            for i in xrange(self.replay_size):
                st_replay[i] = self.D[0][replay_index[i][0]]
                act_replay[i] = self.D[1][replay_index[i][0]]
                r_replay[i] = self.D[2][replay_index[i][0]]
                st_dash_replay[i] = self.D[3][replay_index[i][0]]
                ep_end_replay[i] = self.D[4][replay_index[i][0]]
            #  print "st_replay : ", st_replay
            #  print "act_replay : ", act_replay
            #  print "r_replay : ", r_replay
            #  print "st_dash_replay : ", st_dash_replay
            #  print "ep_end_replay : ", ep_end_replay
            
            self.model.cleargrads()
            #  print "self.model.parameters : ", self.model.parameters
            #  print "self.model.gradients : ", self.model.gradients
            loss, _ = self.forward(st_replay, act_replay, r_replay, st_dash_replay, ep_end_replay)
            loss.backward()
            self.optimizer.update()
            #  print "self.model.gradients(after) : ", self.model.gradients
            #  print "self.model.parameters(after) : ", self.model.parameters

    def train(self, t):
        if self.initial_exploration < t:
            self.experience_replay(t)
            self.reduce_epsilon()
        if t % self.target_update_freq == 0:
            self.target_model = copy.deepcopy(self.model)
        self.step += 1

    def get_action(self, st, evaluation=False):
        if not evaluation:
            if np.random.rand() < self.epsilon:
                return np.random.randint(0, self.n_act), 0
            else:
                if self.gpu >= 0:
                    st = cuda.to_gpu(st)
                s = Variable(st)
                Q = self.model(s)
                Q = Q.data[0]
                a = np.argmax(Q)
                #  print "a : ", int(a)
                #  print "type(a) : ", type(int(a))
                return int(a), max(Q)
        else:
            if arg_gpu >= 0:
                st = cuda.to_gpu(st)
            s = Variable(st)
            Q = self.model(s)
            Q = Q.data[0]
            a = np.argmax(Q)
            #  print "a : ", int(a)
            #  print "type(a) : ", type(int(a))
            return int(a), max(Q)

    def reduce_epsilon(self):
        if self.epsilon > self.epsilon_min:
            #  print "self.epsilon : ", self.epsilon
            self.epsilon -= self.epsilon_decay
        else:
            #  print "self.epsilon___ : ", self.epsilon
            self.epsilon = self.epsilon_min

    def save_model(self, model_dir):
        #  print "Now Saving!!!!!"
        serializers.save_npz(model_dir+"new_model.model", self.model)



if __name__ == "__main__":
    import gym
    env = gym.make('Pendulum-v0')
    
    gpu = 0
    #  gpu = -1
    seed = 0
    n_state = len(env.observation_space.high)
    print n_state
    action_list = [np.array([a]) for a in [-2.0, 2.0]]
    n_action = len(action_list)
    print n_action

    t = 0

    test_agent = Agent(n_state, n_action, gpu, seed)

    for i in xrange(1000):
        state = env.reset()
        for j in xrange(200):
            state = np.array([state], dtype=np.float32)
            print "state : ", state
            #  action = action_list[0]
            act_i, q =  test_agent.get_action(state)
            action = action_list[act_i]
            print "action : ", action

            next_state, reward, episode_end, _ = env.step(action)
            next_state = np.array([next_state], np.float32)
            print "next_state : ", next_state
            print "reward : ", reward
            
            test_agent.stock_experience(t, state, act_i, reward, next_state, episode_end)
            test_agent.train(t)
            print "test_agent.loss : ", test_agent.loss

            t += 1

            state = next_state

            if episode_end:
                break

        #  if t > test_agent.initial_exploration:
            #  break

    print t 

    print test_agent.D






