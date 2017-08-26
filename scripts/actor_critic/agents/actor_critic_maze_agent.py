#!/usr/bin/env python
#coding:utf-8
"""
迷路探索
   5x5の大きさの迷路を探索する
   状態 : 自身のマップ上の位置(Q_tableはマップ全体に確保される)
   行動 : 上下左右の４パターン(上 : a=0 左 : a=1 下 : a=2 右 : a=3)
   報酬 : ゴールに到達した時
   方策 : ガウス分布
"""

import numpy as np

import math

class Agent:
    ALPHA = 0.1
    GAMMA = 0.99

    def __init__(self, maze, num_action):
        self._maze = maze
        self.num_action = num_action
        self.action_list = np.arange(self.num_action, dtype=np.int32)
        self.min_action = self.action_list[0]
        self.max_action = self.action_list[self.num_action-1]
        #  print "self.action_list : ", self.action_list

        self._v_table = np.zeros(len(self._maze[0])*len(self._maze), dtype=np.float32)
        self._mu_table = np.zeros(len(self._maze[0])*len(self._maze), dtype=np.float32)
        self._sigma_table = np.zeros(len(self._maze[0])*len(self._maze), dtype=np.float32)
        self._mu_table = self._mu_table + self.num_action/2.0
        #  print "self._mu_table : ", self._mu_table
        self._sigma_table = self._sigma_table + self.num_action/2.0
        #  print "self._sigma_table : ", self._sigma_table

        self.orig_action = 0

    def state2index(self, state):
        return state[0]*len(self._maze[0]) + state[1]

    def actor(self, state, evaluation=False):
        if not evaluation:
            index = self.state2index(state)
            action = np.random.normal(self._mu_table[index], self._sigma_table[index])
            #  print "action(random) : ", action
            self.orig_action = action

            if action < self.min_action:
                while action < self.min_action:
                    #  print "action(min) : ", action
                    action += self.num_action
                    #  print "action(min)_ : ", action

            if action >= self.max_action+1:
                while action >= self.max_action+1:
                    #  print "action(max) : ", action
                    action -= self.num_action
                    #  print "action(max)_ : ", action

            return int(action)
        else:
            index = self.state2index(state)
            action = self._mu_table[index]

            if action < self.min_action:
                while action < self.min_action:
                    #  print "action(min) : ", action
                    action += self.num_action
                    #  print "action(min)_ : ", action

            if action >= self.max_action+1:
                while action >= self.max_action+1:
                    #  print "action(max) : ", action
                    action -= self.num_action
                    #  print "action(max)_ : ", action

            return int(action)

    """
    criticにはマイナスの報酬を与えない方がいいっぽい...
    状態価値のTD誤差から方策を更新(平均と分散を更新)しているからマイナスの報酬によってマイナス方向に状態価値が
    更新されてしまうと、うれしくない方向に方策が更新されてしまう可能性が生じてしまうことがあるから...
    """
    def critic(self, state, next_state, reward):
        if reward >= 0:
            index = self.state2index(state)
            next_index = self.state2index(next_state)
            td_error = reward + self.GAMMA*self._v_table[next_index] - self._v_table[index]
            self._v_table[index] = self._v_table[index] + self.ALPHA*td_error
            #  print "self._v_table[index] : ", self._v_table[index]
            return td_error
        else:
            return reward

    def mu_update(self, state, action):
        index = self.state2index(state)
        self._mu_table[index] = (self._mu_table[index]+self.orig_action) / 2.0
        #  print "self._mu_table[index] : ", self._mu_table[index]

        if self._mu_table[index] < self.min_action:
            while self._mu_table[index] < self.min_action:
                self._mu_table[index] += self.num_action
        
        if self._mu_table[index] >= self.max_action+1:
            while self._mu_table[index] >= self.max_action+1:
                self._mu_table[index] -= self.num_action

    def sigma_update(self, state, action):
        index = self.state2index(state)
        self._sigma_table[index] = (self._sigma_table[index]+math.fabs(self.orig_action-self._mu_table[index])) / 2.0
        #  print "self._sigma_table[index] : ", self._sigma_table[index]

    def train(self, state, action, next_state, reward, episode_end):
        td_error = self.critic(state, next_state, reward)
        if td_error > 0:
            self.mu_update(state, action)
            self.sigma_update(state, action)

    def show_policy(self):
        action_display_list = ['^', '<', 'v', '>']
        for i in xrange(len(self._maze)):
            for j in xrange(len(self._maze[0])):
                state = [i, j]
                if self._maze[i][j] == 0:
                    action = self.actor(state, evaluation=True)
                    print '%2s' % action_display_list[action],
                else:
                    print '%2d' % self._maze[i][j],
            print 

    def save_tables(self, dir_path):
        pass


if __name__ == "__main__":
    rows = 5
    cols = 5

    test_maze = np.zeros((rows+2, cols+2), dtype=np.int32)

    num_action = 4

    test_agent = Agent(test_maze, num_action)
    
    state = [1, 1]

    action = test_agent.actor(state)
    print "action : ", action
    
    next_state = [1, 2]
    reward = 100
    episode_end = False
    td_error = test_agent.critic(state, next_state, reward)
    print "td_error : ", td_error
    
    #  test_agent.mu_update(state, action)
    #  test_agent.sigma_update(state, action)

    test_agent.train(state, action, next_state, reward, episode_end)
    



