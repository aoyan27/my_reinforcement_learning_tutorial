#!/usr/bin/env python
#coding: utf-8
"""
倒立振子
    倒立振子の振り上げ動作の学習
    状態 : 角度、角速度 -->[cos(theta), sine(theta), theta_dot]
    行動 : トルク--> (-1, 1)
    報酬 : 倒立振子が頂点に来た時を原点として現在の状態のなす角を元に以下の式で算出される
        reward = theta**2 + 0.1 * theta_dot**2 + 0.001*max_toruque**2

    連続な状態空間に対応するための関数近似としてニューラルネットを利用しただけ...
    DQNの特徴であるExperience Replay, Fixed target Q-Network, Reward clipingは使用していない...
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
            l2=L.Linear(100, 100),
            l3=L.Linear(100, 100),
            l4=L.Linear(100, n_action, initialW=np.zeros((n_action, 100), dtype=np.float32)),
        )
        

    def __call__(self, state, gpu=-1):
        if gpu >= 0:
            state = cuda.to_gpu(state)
        x = Variable(state)
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        y = self.l4(h)
        return y

class Agent:
    ALPHA = 1e-6
    GAMMA = 0.9

    def __init__(self, n_state, n_action, gpu):
        self.n_state = n_state
        self.n_action = n_action
        self.action_list = np.arange(-1, 2, 1)

        self.gpu = gpu

        self.model = ActionValueNetwork(self.n_state, self.n_action)
        if self.gpu >= 0:
            self.model.to_gpu()

        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

        #  self.epsilon = 1.0
        self.epsilon = 0.1
        self.epsilon_decay = 0.00001
        self.min_epsilon = 0.1

    def forward(self, state, next_state, action, reward, episode_end):
        Q = self.model(state, self.gpu)
        #  print "Q : ", Q.data
        tmp = self.model(next_state, self.gpu)
        tmp = map(np.max, tmp.data)
        #  print "tmp : ", tmp
        max_next_Q = np.array(tmp, dtype=np.float32)
        #  print "max_next_Q : ", max_next_Q
        target = copy.deepcopy(Q.data)
        #  print "target : ", target

        if not episode_end:
            tmp_ = reward + self.GAMMA*max_next_Q
        else:
            tmp_ = reward
        #  print "tmp_ : ", tmp_
        #  print "action : ", action
        #  print "np.where(self.action_list == action) : ", np.where(self.action_list == action)
        action_index = np.where(self.action_list == action)[0][0]
        #  print "action_index : ", action_index
        #  print "target : ", target[0][action_index]
        target[0][action_index] = tmp_[0]
        #  print "target : ", target

        td = Variable(target) - Q
        #  print "td : ", td.data

        zero_val = np.zeros((1, self.n_action), dtype=np.float32)
        if self.gpu >= 0:
            zero_val = cuda.to_gpu(zero_val)
        zero_val = Variable(zero_val)
        #  print "zero_val : ", zero_val.data

        loss = F.mean_squared_error(td, zero_val)
        #  print "loss : ", loss.data

        return loss, Q

    def train(self, state, next_state, action, reward, episode_end):
        self.model.cleargrads()
        loss, _ = self.forward(state, next_state, action, reward, episode_end)
        loss.backward()
        self.optimizer.update()

    def get_action(self, state, evaluation=False):
        if not evaluation:
            if np.random.rand() < self.epsilon:
                #  print "Rondom action!!"
                return self.action_list[np.random.randint(0, self.n_action)], 0
            else:
                Q = self.model(state, self.gpu)
                #  print "Q(action) : ", Q.data
                Q = Q.data[0]
                #  print "Q(action) : ", Q
                action = self.action_list[int(np.argmax(Q))]
                #  print "action : ", action
                return action, Q.max()
        else:
            Q = self.model(state, self.gpu)
            #  print "Q(action) : ", Q.data
            Q = Q.data[0]
            #  print "Q(action) : ", Q
            action = self.action_list[np.argmax(Q)]
            #  print "action : ", action
            return action, Q.max()
    
    def save_model(self, path):
        if self.gpu >= 0:
            self.model.to_cpu()
        print "Save model to {}".format(path)
        serializers.save_npz(path, self.model)


if __name__ == "__main__":
    import gym
    env = gym.make('Pendulum-v0')
    num_state = len(env.observation_space.high)
    num_action = 3
    gpu = -1

    test_agent = Agent(num_state, num_action, gpu)
    action_list = np.array([-1, 0, 1], dtype=np.int32)
    np.random.shuffle(action_list)
    action = np.array([action_list[0]])
    print "action : ", action
    state = env.reset()
    print "state : ", state
    next_state, reward, episode_end, _ = env.step(action)
    print "next_state : ", next_state

    state = np.array([state], dtype=np.float32)
    next_state = np.array([next_state], dtype=np.float32)

    test_agent.forward(state, next_state, action, reward, episode_end)

    test_agent.train(state, next_state, action, reward, episode_end)

    action, Q_max = test_agent.get_action(state)
    print "action : ", action, " Q_max : ", Q_max
