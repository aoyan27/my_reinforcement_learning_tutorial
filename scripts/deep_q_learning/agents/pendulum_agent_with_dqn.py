#!/usr/bin/env python
#coding: utf-8
"""
倒立振子
    倒立振子の振り上げ動作の学習
    状態 : 角度、角速度 -->[cos(theta), sine(theta), theta_dot]
    行動 : トルク--> (-1, 1)
    報酬 : 倒立振子が頂点に来た時を原点として現在の状態のなす角を元に以下の式で算出される
        reward = theta**2 + 0.1 * theta_dot**2 + 0.001*max_toruque**2
    DQNを参考にしている(Experience Replay, Fixed Q-Network, Reward Clipping)
"""

import numpy as np

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I

import copy

class ActionValueNetwork(Chain):
    def __init__(self, n_state, n_action):
        super(ActionValueNetwork, self).__init__(
            l1=L.Linear(n_state, 100),
            #  l2=L.Linear(100, 100),
            #  l3=L.Linear(100, 100),
            l4=L.Linear(100, n_action, initialW=np.zeros((n_action, 100), dtype=np.float32)),
        )
        

    def __call__(self, state):
        x = Variable(state)
        h = F.relu(self.l1(x))
        #  h = F.relu(self.l2(h))
        #  h = F.relu(self.l3(h))
        y = self.l4(h)
        return y

class Agent:
    #  ALPHA = 1e-6
    ALPHA = 0.1
    GAMMA = 0.9

    MAX_ACTION = 2.0
    MIN_ACTION = -1.0 * MAX_ACTION

    def __init__(self, n_state, n_action, gpu):
        self.n_state = n_state
        self.n_action = n_action

        self.gpu = gpu

        self.model = ActionValueNetwork(self.n_state, self.n_action)
        self.target_model = copy.deepcopy(self.model)
        if self.gpu >= 0:
            self.model.to_gpu()
            self.target_model.to_gpu()
        
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)



        self.action_list = np.linspace(self.MIN_ACTION, self.MAX_ACTION, self.n_action)
        self.action_list = [np.array([a]) for a in self.action_list]

        self.epsilon = 1.0
        self.epsilon_decay = 0.000001
        self.epsilon_min = 0.1
        
        self.data_size = 1000
        self.replay_size = 100

        self.init_exprolation = 1000
        self.target_update_freq = 10000

        # self.D = [state, action, next_state, reward, episode_end]
        self.D = self.create_experience_memory()
        #  self.D = [np.zeros((self.data_size, 1, self.n_state), dtype=np.float32),
                  #  np.zeros((self.data_size, 1), dtype=np.float32),
                  #  np.zeros((self.data_size, 1, self.n_state), dtype=np.float32),
                  #  np.zeros((self.data_size, 1), dtype=np.float32),
                  #  np.zeros(self.data_size, dtype=np.bool)]
        
        # self.replay_memory = [state, action, next_state, reward, episode_end] each element size is self.replay_size
        self.replay_memory = self.create_replay_memory()
        #  self.replay_memory = [np.zeros((self.replay_size, 1, self.n_state), dtype=np.float32),
                              #  np.zeros((self.replay_size, 1), dtype=np.float32),
                              #  np.zeros((self.replay_size, 1, self.n_state), dtype=np.float32),
                              #  np.zeros((self.replay_size, 1), dtype=np.float32),
                              #  np.zeros(self.replay_size, dtype=np.bool)]

    def create_experience_memory(self):
        state = np.zeros((self.data_size, 1, self.n_state), dtype=np.float32)
        action = np.zeros((self.data_size, 1), dtype=np.float32)
        next_state = np.zeros((self.data_size, 1, self.n_state), dtype=np.float32)
        reward = np.zeros(self.data_size, dtype=np.float32)
        episode_end = np.zeros(self.data_size, dtype=np.bool)
        if self.gpu >= 0:
            state = cuda.to_gpu(state)
            action = cuda.to_gpu(action)
            next_state = cuda.to_gpu(next_state)
            reward = cuda.to_cpu(reward)
            episode_end = cuda.to_gpu(episode_end)
        
        return [state, action, next_state, reward, episode_end]

    def create_replay_memory(self):
        state = np.zeros((self.replay_size, 1, self.n_state), dtype=np.float32)
        action = np.zeros((self.replay_size, 1), dtype=np.float32)
        next_state = np.zeros((self.replay_size, 1, self.n_state), dtype=np.float32)
        reward = np.zeros(self.replay_size, dtype=np.float32)
        episode_end = np.zeros(self.replay_size, dtype=np.bool)
        if self.gpu >= 0:
            state = cuda.to_gpu(state)
            action = cuda.to_gpu(action)
            next_state = cuda.to_gpu(next_state)
            reward = cuda.to_cpu(reward)
            episode_end = cuda.to_gpu(episode_end)
        
        return [state, action, next_state, reward, episode_end]


    def stock_experience(self, t, state, action, next_state, reward, episode_end):
        index = t % self.data_size
        if self.gpu >= 0:
            state = cuda.to_gpu(state)
            action = cuda.to_gpu(action)
            next_state = cuda.to_gpu(next_state)
            reward = cuda.to_gpu(reward)
            episode_end = cuda.to_gpu(episode_end)
        self.D[0][index] = state
        self.D[1][index] = action
        self.D[2][index] = next_state
        self.D[3][index] = reward
        self.D[4][index] = episode_end

    def forward(self):
        num_batch = self.replay_size
        Q = self.model(self.replay_memory[0])
        #  print "Q.data_ : ", Q.data
        tmp = self.target_model(self.replay_memory[2])
        #  tmp = self.model(self.replay_memory[2])
        #  print "tmp.data : ", tmp.data
        tmp = map(np.max, tmp.data)
        #  print type(tmp)
        max_next_Q = np.array(tmp)
        #  print "max_next_Q : ", max_next_Q

        target = copy.deepcopy(Q.data)
        #  print "target : ", Q.data

        for i in xrange(num_batch):
            if self.replay_memory[4][i]:
                tmp_ = self.replay_memory[3][i] + self.GAMMA*max_next_Q[i]
            else:
                tmp_ = self.replay_memory[3][i]
            #  print tmp_
            action_index = 0
            if self.replay_memory[1][i] == -2:
                action_index = 0
            else:
                action_index = 1

            #  action_index = np.where(self.action_list == self.replay_memory[1][i])[0][0]
            #  print "action : ", self.replay_memory[1][i]
            #  print "self.action_list[0] : ", self.action_list[0]
            #  print "self.action_list : ", self.action_list
            #  print "action_index : ", action_index
            target[i][action_index] = tmp_
        #  print "target : ", target

        td = Variable(target) - Q
        #  print "td.data : ", td.data

        zero_val = np.zeros((num_batch, self.n_action), dtype=np.float32)
        if self.gpu >= 0:
            zero_val = cuda.to_gpu(zero_val)
        zero_val = Variable(zero_val)

        loss = F.mean_squared_error(td, zero_val)
        #  print "loss.data : ", loss.data

        return loss, Q

    def extract_replay_memory(self):
        index_list = np.arange(self.data_size)
        #  print "index_list : ", index_list
        replay_index_list = np.random.choice(index_list, self.replay_size, replace=False)
        #  print "replay_index_list : ", replay_index_list
        self.replay_memory[0] = self.D[0][replay_index_list][:]
        self.replay_memory[1] = self.D[1][replay_index_list][:]
        self.replay_memory[2] = self.D[2][replay_index_list][:]
        self.replay_memory[3] = self.D[3][replay_index_list][:]
        self.replay_memory[4] = self.D[4][replay_index_list]

    def train(self):
        self.model.cleargrads()
        loss, Q = self.forward()
        #  print "Q.data : ", Q.data
        loss.backward()
        self.optimizer.update()

    def get_action(self, state, evaluation=False):
        if not evaluation:
            if np.random.rand() < self.epsilon:
                #  print "Rondom action!!"
                return self.action_list[np.random.randint(0, self.n_action)], 0
            else:
                if self.gpu >= 0:
                    state = cuda.to_gpu(state)
                Q = self.model(state)
                #  print "Q(action) : ", Q.data
                Q = Q.data[0]
                #  print "Q(action) : ", Q
                action = self.action_list[int(np.argmax(Q))]
                #  print "action : ", action
                return action, Q.max()
        else:
            if self.gpu >= 0:
                state = cuda.to_gpu(state)
            Q = self.model(state)
            #  print "Q(action) : ", Q.data
            Q = Q.data[0]
            #  print "Q(action) : ", Q
            action = self.action_list[np.argmax(Q)]
            #  print "action : ", action
            return action, Q.max()

    def reduce_epsilon(self):
        if self.epsilon > self.epsilon_min:
            #  print "self.epsilon : ", self.epsilon
            self.epsilon -= self.epsilon_decay
        else:
            #  print "self.epsilon___ : ", self.epsilon
            self.epsilon = self.epsilon_min

    def target_model_update(self, t):
        if t%self.target_update_freq == 0:
            self.target_model = copy.deepcopy(self.model)
    
    def save_model(self, path):
        if self.gpu >= 0:
            self.model.to_cpu()
        print "Save model to {}".format(path)
        serializers.save_npz(path, self.model)


if __name__ == "__main__":
    import gym
    env = gym.make('Pendulum-v0')

    n_state = len(env.observation_space.high)
    n_action = 2

    gpu = -1
    test_agent = Agent(n_state, n_action, gpu)
    
    t = 0
    for i_episode in xrange(100):
        state = env.reset()
        for j_step in xrange(200):
            #  action = env.action_space.sample()
            #  if action > 0:
                #  print "+2.0"
                #  action = np.array([2.0])
            #  else:
                #  print "-2.0"
                #  action = np.array([-2.0])
            state = np.array([state], dtype=np.float32)
            action, _ = test_agent.get_action(state)
            
            next_state, reward, done, info = env.step(action)
            next_state = np.array([next_state], dtype=np.float32)
            
            print "t : ", t
            test_agent.stock_experience(t, state, action, next_state, reward, done)
            
            print "state : {0} action : {1} next_state : {2} reward : {3} done : {4}".format(state, action, next_state, reward, done)

            if t > test_agent.init_exprolation:
                test_agent.extract_replay_memory()
                loss = test_agent.forward()

            if done:
                break
            else:
                state = next_state


            t += 1

        if t > test_agent.init_exprolation:
            break
    
    #  print type(test_agent.D[0][0])
    #  print type(state)
    #  test_agent.D[0][0] = state
    #  print test_agent.D[0][0]
    #  print test_agent.model(np.array([state], dtype=np.float32)).data
