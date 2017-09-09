#!/usr/bin/env python
#coding:utf-8

import numpy as np
import sys
import copy
import time
import math

import chainer
from chainer import cuda, Variable, optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

from random import random, randint

class criticNetwork(Chain):
    def __init__(self, n_in, n_out):
        super(criticNetwork, self).__init__(
                l1 = L.Linear(n_in, 100),
                l2 = L.Linear(100, 200),
                l3 = L.Linear(200, 100),
                l4 = L.Linear(100, 100),
                l5 = L.Linear(100, n_out, initialW=np.zeros((n_out, 100), dtype=np.float32)),
                )

    def v_func(self, x):
        h = F.leaky_relu(self.l1(x))
        h = F.leaky_relu(self.l2(h))
        h = F.leaky_relu(self.l3(h))
        h = F.leaky_relu(self.l4(h))
        y = self.l5(h)
        return y


class actorNetwork(Chain):
    def __init__(self, n_in, n_out):
        super(actorNetwork, self).__init__(
                l1 = L.Linear(n_in, 100),
                l2 = L.Linear(100, 200),
                l3 = L.Linear(200, 100),
                l4 = L.Linear(100, 100),
                l5 = L.Linear(100, n_out, initialW=np.zeros((n_out, 100), dtype=np.float32)),
                )

    def actor_func(self, x):
        h = F.leaky_relu(self.l1(x))
        h = F.leaky_relu(self.l2(h))
        h = F.leaky_relu(self.l3(h))
        h = F.leaky_relu(self.l4(h))
        y = self.l5(h)
        return y


class Agent:
    GAMMA = 0.99

    data_size = 1000
    replay_size = 100
    initial_exploration = 1000
    target_update_freq = 20

    def __init__(self, n_state, n_action, gpu, seed):
        np.random.seed(seed)
        sys.setrecursionlimit(10000)
        self.n_state = n_state
        self.n_action = n_action

        self.gpu = gpu

        self.critic_model = criticNetwork(self.n_state, self.n_action)
        self.target_critic_model = copy.deepcopy(self.critic_model)
        self.actor_model = actorNetwork(self.n_state, self.n_action)
        if self.gpu >= 0:
            self.critic_model.to_gpu()
            self.target_critic_model.to_gpu()
            self.actor_model.to_gpu()

        self.critic_optimizer = optimizers.Adam()
        self.critic_optimizer.setup(self.critic_model)
        self.actor_optimizer = optimizers.Adam()
        self.actor_optimizer.setup(self.actor_model)

        self.D_critic = self.create_history_memory()
        self.D_actor = self.create_history_memory()

        #  self.D_critic = [np.zeros((self.data_size, 1, self.n_state), dtype=np.float32),
                         #  np.zeros((self.data_size, self.n_action), dtype=np.float32),
                         #  np.zeros((self.data_size, 1, self.n_state), dtype=np.float32),
                         #  np.zeros(self.data_size, dtype=np.float32),
                         #  np.zeros(self.data_size, dtype=np.bool)]

        #  self.D_actor = [np.zeros((self.data_size, 1, self.n_state), dtype=np.float32),
                         #  np.zeros((self.data_size, self.n_action), dtype=np.float32),
                         #  np.zeros((self.data_size, 1, self.n_state), dtype=np.float32),
                         #  np.zeros(self.data_size, dtype=np.float32),
                         #  np.zeros(self.data_size, dtype=np.bool)]

        self.critic_loss = 0.0
        self.actor_loss = 0.0
        self.step = 0

        self.data_index_actor = 0
        
        ### Pendulum-v0 ###
        #  self.min_action = -2.0
        #  self.max_action = 2.0
        #  self.limit_action = 2.0

        ### MountainCarContinuous-v0, InvertedPendulum-v1 ###
        self.min_action = -1.0
        self.max_action = 1.0
        self.limit_action = 1.0
        

        self.sigma = 1.0
        self.sigma_decay = 1e-6
        self.sigma_min = 0.1

        self.V = 0.0

    def create_history_memory(self):
        state = np.zeros((self.data_size, 1, self.n_state), dtype=np.float32)
        action = np.zeros((self.data_size, self.n_action), dtype=np.float32)
        next_state = np.zeros((self.data_size, 1, self.n_state), dtype=np.float32)
        reward = np.zeros(self.data_size, dtype=np.float32)
        episode_end = np.zeros(self.data_size, dtype=np.bool)

        if self.gpu >= 0:
            state = cuda.to_gpu(state)
            action = cuda.to_gpu(action)
            next_state = cuda.to_gpu(next_state)
            reward = cuda.to_gpu(reward)
            episode_end = cuda.to_gpu(episode_end)
        return [state, action, next_state, reward, episode_end]

    def stock_experience_for_critic(self, t, state, action, next_state, reward, episode_end):
        data_index = t % self.data_size
        if self.gpu >= 0:
            state = cuda.to_gpu(state)
            action = cuda.to_gpu(action)
            next_state = cuda.to_gpu(next_state)
            reward = cuda.to_gpu(reward)
            episode_end = cuda.to_gpu(episode_end)

        self.D_critic[0][data_index] = state
        self.D_critic[1][data_index] = action
        self.D_critic[2][data_index] = next_state
        self.D_critic[3][data_index] = reward
        self.D_critic[4][data_index] = episode_end

    def critic_forward(self, state, action, next_state, reward, episode_end):
        num_batch = state.shape[0]

        s = Variable(state)
        s_dash = Variable(next_state)

        V = self.critic_model.v_func(s)
        tmp = self.target_critic_model.v_func(s_dash)

        target = tmp.data

        for i in xrange(num_batch):
            if not episode_end[i]:
                tmp_ = reward[i] + self.GAMMA*tmp.data[i]
            else:
                tmp_ = reward[i]

            target[i][0] = tmp_
        
        #  td = Variable(target) - V
        #  zero_val = Variable(np.zeros((self.replay_size, self.n_action), dtype=np.float32))

        loss = F.mean_squared_error(V, Variable(target))
        #  loss = F.mean_squared_error(td, zero_val)
        self.critic_loss = loss.data

        return loss, V

    def critic_experience_replay(self, t):
        if t > self.initial_exploration:
            index_list = np.arange(self.data_size)
            replay_index_list = np.random.permutation(index_list)
            index = replay_index_list[0:self.replay_size]

            state_replay = self.D_critic[0][index]
            action_replay = self.D_critic[1][index]
            next_state_replay = self.D_critic[2][index]
            reward_replay = self.D_critic[3][index]
            episode_end_replay = self.D_critic[4][index]

            self.critic_model.zerograds()
            #  self.model.cleargrads()
            loss, _ = self.critic_forward(state_replay, action_replay, next_state_replay, reward_replay, episode_end_replay)
            loss.backward()
            self.critic_optimizer.update()

    def calculate_td_error(self, state, next_state, reward):
        state = state.astype(np.float32)
        next_state = next_state.astype(np.float32)
        
        ### Pendulum-v0 ###
        #  reward = reward.astype(np.float32)
        
        ### MountainCarContinuous-v0 ###
        reward = np.array([reward], dtype=np.float32)
        #  print "reward : ", reward




        if self.gpu >= 0:
            state = cuda.to_gpu(state)
            next_state = cuda.to_gpu(next_state)
            reward = cuda.to_gpu(reward)

        s = Variable(state)
        s_dash = Variable(next_state)

        V = self.critic_model.v_func(s)
        self.V = V.data
        tmp = self.target_critic_model.v_func(s_dash)

        target = reward + self.GAMMA*tmp.data
        
        td = Variable(target) - V

        return td.data[0][0]

    def stock_experience_for_actor(self, t, state, action, next_state, reward, episode_end):
        if t > self.initial_exploration:
            if self.calculate_td_error(state, next_state, reward) > 0.0:
                data_index = self.data_index_actor % self.data_size
                if self.gpu >= 0:
                    state = cuda.to_gpu(state)
                    action = cuda.to_gpu(action)
                    next_state = cuda.to_gpu(next_state)
                    reward = cuda.to_gpu(reward)
                    episode_end = cuda.to_gpu(episode_end)

                self.D_actor[0][data_index] = state
                self.D_actor[1][data_index] = action
                self.D_actor[2][data_index] = next_state
                self.D_actor[3][data_index] = reward
                self.D_actor[4][data_index] = episode_end

                self.data_index_actor += 1
    
    def actor_forward(self, state, action, next_state, reward, episode_end):
        num_batch = state.shape[0]

        s = Variable(state)
        #  print "s.data : ", s.data
        a = self.actor_model.actor_func(s)
        #  print "a.data : ", a.data

        target = Variable(action)
        #  print "target.data : ", target.data

        loss = F.mean_squared_error(a, target)
        self.actor_loss = loss.data

        return loss, a

    def actor_experience_replay(self, t):
        if self.data_index_actor >  self.initial_exploration:
            index_list = np.arange(self.data_size)
            replay_index_list = np.random.permutation(index_list)
            index = replay_index_list[0:self.replay_size]

            state_replay = self.D_actor[0][index]
            action_replay = self.D_actor[1][index]
            next_state_replay = self.D_actor[2][index]
            reward_replay = self.D_actor[3][index]
            episode_end_replay = self.D_actor[4][index]

            self.actor_model.zerograds()
            loss, _ = self.actor_forward(state_replay, action_replay, next_state_replay, reward_replay, episode_end_replay)
            loss.backward()
            self.actor_optimizer.update()


    def train(self, t, state, action, next_state, reward, episode_end):
        self.stock_experience_for_critic(t, state, action, next_state, reward, episode_end)
        self.stock_experience_for_actor(t, state, action, next_state, reward, episode_end)

        self.critic_experience_replay(t)
        self.actor_experience_replay(t)

        self.reduce_sigma()
        if t % self.target_update_freq == 0:
            self.target_critic_model = copy.deepcopy(self.critic_model)

        self.step += 1


    def BoxMuller(self, mean, var):
        r1 = random()
        r2 = random()

        z1 = math.sqrt(-2.0 * math.log(r1))
        z2 = math.sin(2.0 * math.pi * r2)

        return var * z1 * z2 + mean

    def get_action(self, state, evaluation=False):
        #  print "st(get_action) : ", st, type(st)
        if self.gpu >= 0:
            state = cuda.to_gpu(state)
        s = Variable(state)
        a = self.actor_model.actor_func(s)
        
        if not evaluation:
            action = a.data[0][0] + self.BoxMuller(0.0, self.sigma)
            if action < self.min_action:
                while 1:
                    action += self.limit_action
                    if action > self.min_action:
                        break

            if action >= self.max_action:
                while 1:
                    action -= self.limit_action
                    if action <= self.max_action:
                        break
        else:
            action = a.data[0][0]
        
        action = action.reshape((1, self.n_action))
        a_ = a.data[0][0]
        if self.gpu >= 0:
            action = cuda.to_cpu(action)
            a_ = cuda.to_cpu(a_)

        return action, a_

    def reduce_sigma(self):
        if self.sigma > self.sigma_min:
            #  print "self.sigma : ", self.sigma
            self.sigma -= self.sigma_decay
        else:
            #  print "self.sigma_ : ", self.sigma
            self.sigma = self.sigma_min
    
    def save_model(self, model_dir):
        serializers.save_npz(model_dir+'critic_model.model', self.critic_model)
        serializers.save_npz(model_dir+'actor_model.model', self.actor_model)

    def load_model(self, model_dir):
        serializers.load_npz(model_dir+'critic_model.model', self.critic_model)
        serializers.load_npz(model_dir+'actor_model.model', self.actor_model)





if __name__ == "__main__":
    import gym
    env = gym.make('Pendulum-v0')

    n_state = env.observation_space.shape[0]
    n_action = 1
    
    print n_state, n_action
    
    #  gpu = -1
    gpu = 0
    seed = 0

    agent = Agent(n_state, n_action, gpu, seed)
    
    t = 0

    for i in xrange(100):
        observation = env.reset()
        for j in xrange(2000):
            state = observation.astype(np.float32).reshape((1, n_state))
            print "state : ", state, type(state)

            #  action = env.action_space.sample()
            action, a = agent.get_action(state)
            print "action : ", action, type(state)
            print "a : ", a

            observation, reward, episode_end, _ = env.step(action)
            next_state = observation.astype(np.float32).reshape((1, n_state))
            print "next_state : ", next_state, type(next_state)

            agent.train(t, state, action, next_state, reward, episode_end)

            print "step : ", agent.step, "\tdata_index_actor : ", agent.data_index_actor

            t += 1


            if episode_end:
                break




