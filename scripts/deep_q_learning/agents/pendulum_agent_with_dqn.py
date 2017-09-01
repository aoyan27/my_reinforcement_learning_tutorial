#!/usr/bin/env python
#coding: utf-8
"""
倒立振子
    倒立振子の振り上げ動作の学習
    状態 : 角度、角速度 -->[cos(theta), sine(theta), theta_dot]
    行動 : トルク--> (-1, 1)
    報酬 : 倒立振子が頂点に来た時を原点として現在の状態のなす角を元に以下の式で算出される
        reward = -1 * (theta**2 + 0.1 * theta_dot**2 + 0.001*max_toruque**2)
    DQNを参考にしている(Experience Replay, Fixed Q-Network, Reward Clipping)
"""

import numpy as np

import chainer
from chainer import cuda, Variable, optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

import copy

class ActionValueNetwork(Chain):
    def __init__(self, n_state, n_action):
        super(ActionValueNetwork, self).__init__(
            l1=L.Linear(n_state, 100),
            l2=L.Linear(100, 200),
            l3=L.Linear(200, 100),
            l4=L.Linear(100, 100),
            l5=L.Linear(100, n_action, initialW=np.zeros((n_action, 100), dtype=np.float32)),
        )

    def __call__(self, x):
        h = F.leaky_relu(self.l1(x))
        h = F.leaky_relu(self.l2(h))
        h = F.leaky_relu(self.l3(h))
        h = F.leaky_relu(self.l4(h))
        y = self.l5(h)
        return y

class Agent:
    ALPHA = 0.1
    GAMMA = 0.9

    data_size = 100
    replay_size = 100
    init_exprolation = 100
    target_update_freq = 20

    def __init__(self, n_state, n_action, gpu):
        self.n_state = n_state
        self.n_action = n_action

        self.gpu = gpu

        self.model = ActionValueNetwork(self.n_state, self.n_action)
        self.target_model = copy.deepcopy(self.model)
        if self.gpu >= 0:
            self.model.to_gpu()
            self.target_model.to_gpu()
        
        #  self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, eps=0.0001)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

        self.D = self.create_history_memory()

        self.epsilon = 1.0
        self.epsilon_decay = 0.00001
        self.min_epsilon = 0.05

        self.loss = 0.0
        self.step = 0

    def create_history_memory(self):
        state = np.zeros((self.data_size, 1, self.n_state), dtype=np.float32)
        action = np.zeros(self.data_size, dtype=np.float32)
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

    def forward(self, state, action, next_state, reward, episode_end):
        xp = cuda.get_array_module(state)
        num_batch = state.shape[0]
        #  print "num_batch : ", num_batch, type(num_batch)

        state = Variable(state)
        next_state = Variable(next_state)

        Q = self.model(state)
        #  print "Q.data : ", Q.data, type(Q.data)
        tmp = self.target_model(next_state)
        #  print "tmp.data : ", tmp.data, type(tmp.data)

        tmp = map(xp.max, tmp.data)
        #  print "tmp : ", tmp, type(tmp)
        max_Q_dash = xp.array(tmp, dtype=xp.float32)
        #  print "max_Q_dash : ", max_Q_dash, type(max_Q_dash)
        
        target = copy.deepcopy(Q.data)
        #  print "target : ", target, type(target)

        for i in xrange(num_batch):
            #  print "reward[i] : ", reward[i], type(reward[i])
            #  print "max_Q_dash[i] : ", max_Q_dash[i], type(max_Q_dash[i])
            if not episode_end[i]:
                tmp_ = reward[i] + self.GAMMA*max_Q_dash[i]
                #  print "tmp_(not episode_end) : ", tmp_, type(tmp_)
            else:
                tmp_ = reward[i]
                #  print "tmp_(episode_end) : ", tmp_, type(tmp_)
            #  print "action[i] : ", action[i], type(action[i])
            action_index = int(action[i])
            #  print "action_index : ", action_index, type(action_index)
            target[i][action_index] = tmp_

        #  print "target(after) : ", target

        td = Variable(target) - Q
        #  print "td.data : ", td.data

        zero_val = Variable(xp.zeros((num_batch, self.n_action), dtype=xp.float32))
        #  print "zero_val.data : ", zero_val.data, type(zero_val.data)

        loss = F.mean_squared_error(td, zero_val)
        #  print "loss.data : ", loss.data

        self.loss = loss.data

        return loss, Q

    def experience_replay(self):
        index_list = np.arange(self.data_size)
        #  print "index_list : ", index_list
        replay_index_list = np.random.choice(index_list, self.replay_size, replace=False)
        #  print "replay_index_list : ", replay_index_list 

        state_replay = self.D[0][replay_index_list][:]
        #  print "state_replay : ", state_replay, type(state_replay)
        action_replay = self.D[1][replay_index_list]
        #  print "action_replay : ", action_replay, type(action_replay)
        next_state_replay = self.D[2][replay_index_list][:]
        #  print "next_state_replay : ", next_state_replay, type(next_state_replay)
        reward_replay = self.D[3][replay_index_list]
        #  print "reward_replay : ", reward_replay, type(reward_replay)
        episode_end_replay = self.D[4][replay_index_list]
        #  print "episode_end_replay : ", episode_end_replay, type(episode_end_replay)

        self.model.cleargrads()
        loss, _ = self.forward(state_replay, action_replay, next_state_replay, reward_replay, episode_end_replay)
        loss.backward()
        self.optimizer.update()

    def get_action(self, state, evaluation=False):
        if self.gpu >= 0:
            state = cuda.to_gpu(state)

        if not evaluation:
            if np.random.rand() < self.epsilon:
                #  print "Random!!"
                action = np.random.randint(0, self.n_action)
                #  print "action : ", action, type(action)
                return action, 0
            else:
                #  print "Greedy!!"
                state = Variable(state)
                #  print "state.data : ", state.data
                Q = self.model(state)
                action = np.argmax(Q.data)
                #  print "action : ", action, type(action)
                return int(action), np.max(Q.data)
        else:
            print "Greedy!!!"
            state = Variable(state)
            #  print "state.data : ", state.data
            Q = self.model(state)
            action = np.argmax(Q.data)
            #  print "action : ", action, type(action)
            return int(action), np.max(Q.data)
    
    def reduce_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = self.min_epsilon

    def train(self, t):
        if t > self.init_exprolation:
            self.experience_replay()
            self.reduce_epsilon()

        if t > self.target_update_freq:
            self.target_model = copy.deepcopy(self.model)

        self.step += 1

    def save_model(self, model_dir):
        file_name = model_dir + "model.model"
        serializers.save_npz(file_name, self.model)

    def load_model(self, model_dir):
        file_name = model_dir + "model.model"
        serializers.load_npz(file_name, self.model)


if __name__ == "__main__":
    import gym
    env = gym.make('Pendulum-v0')

    gpu = 0
    #  gpu = -1

    n_state = len(env.observation_space.high)
    #  print "n_state : ", n_state
    action_list = [np.array([a]) for a in [-2.0, 2.0]]
    #  print "action_list : ", action_list
    n_action = len(action_list)
    #  print "n_action : ", n_action

    agent = Agent(n_state, n_action, gpu)
    
    t = 0

    for i_episode in xrange(100):
        state = env.reset()
        for j_step in xrange(200):
            state = np.array([state], dtype=np.float32)

            #  act_i = 0
            #  act_i = 1
            act_i, q = agent.get_action(state, evaluation=False)
            action = action_list[act_i]
            #  action, _ = test_agent.get_action(state)
            
            next_state, reward, done, info = env.step(action)
            next_state = np.array([next_state], dtype=np.float32)
            
            #  print "t : ", t
            agent.stock_experience(t, state, act_i, next_state, reward, done)
            agent.train(t)
            
            print "state : {0} action : {1} next_state : {2} reward : {3} done : {4}".format(state, action, next_state, reward, done)

            #  if t > test_agent.init_exprolation:
                #  test_agent.extract_replay_memory()
                #  loss = test_agent.forward()

            state = next_state

            t += 1

            if done:
                break



        #  if t > test_agent.init_exprolation:
            #  break
    
    #  #  print type(test_agent.D[0][0])
    #  #  print type(state)
    #  #  test_agent.D[0][0] = state
    #  #  print test_agent.D[0][0]
    #  #  print test_agent.model(np.array([state], dtype=np.float32)).data
