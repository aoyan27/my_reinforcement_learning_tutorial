#!/usr/bin/env python
#coding:utf-8
"""
Frozen lake
    4x4('FrozenLake-v0') or 8x8('FrozenLake8x8-v0')の2つの大きさの環境が用意されている
    状態 : マップ上の自身の位置が数字として与えられる(Q_tableはマップ全体に対して確保される)
    行動 : 上下左右の4パターン(確率で選択した方向から+-1ずれることあり、例えば、上を選択すると左右も動作候補として考慮され1/3の確率で選択される)
    報酬 : ゴールに辿り着いたら報酬-->2 それ以外-->0
"""

import numpy as np

import math

class Agent:
    def __init__(self, env, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma
        self.num_state = env.observation_space.n
        self.num_action = env.action_space.n
        self.action_list = np.arange(self.num_action, dtype=np.int32)
        self.min_action = self.action_list[0]
        self.max_action = self.action_list[self.num_action-1]

        self._v_table = np.zeros(self.num_state, dtype=np.float32)
        self._mu_table = np.zeros(self.num_state, dtype=np.float32)
        self._sigma_table = np.zeros(self.num_state, dtype=np.float32)

        self._mu_table = self._mu_table + self.num_action/2.0
        print "self._mu_table : ", self._mu_table
        self._sigma_table = self._sigma_table + self.num_action/2.0
        print "self._sigma_table : ", self._sigma_table

        self.orig_action = 0

    def actor(self, state, evaluation=False):
        if not evaluation:
            action = np.random.normal(self._mu_table[state], self._sigma_table[state])
            self.orig_action = action

            if action < self.min_action:
                while action < self.min_action:
                    action += self.num_action

            if action >= self.max_action+1:
                while action >= self.max_action+1:
                    action -=self.max_action

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

    def critic(self, state, next_state, reward):
        #  print "state : ", state
        #  print "next_state : ", next_state
        #  print "self._v_table[", state, "] : ", self._v_table[state]
        #  print "self._v_table[", next_state, "] : ", self._v_table[next_state]
        td_error = reward + self.gamma*self._v_table[next_state] - self._v_table[state]
        #  print "td_error : ", td_error
        self._v_table[state] = self._v_table[state] + self.alpha*td_error
        #  print "self._v_table[", state, "] : ", self._v_table[state]
        return td_error

    def mu_update(self, state, action):
        self._mu_table[state] = (self._mu_table[state]+self.orig_action) / 2.0

        if self._mu_table[state] < self.min_action:
            while self._mu_table[state] < self.min_action:
                self._mu_table[state] += self.num_action
        
        if self._mu_table[state] >= self.max_action+1:
            while self._mu_table[state] >= self.max_action+1:
                self._mu_table[state] -= self.num_action

    def sigma_update(self, state, action):
        self._sigma_table[state] = (self._sigma_table[state]+math.fabs(self.orig_action-self._mu_table[state])) / 2.0
        #  print "self._sigma_table[state] : ", self._sigma_table[state]

    def train(self, state, action, next_state, reward, episode_end):
        td_error = self.critic(state, next_state, reward)
        if td_error > 0:
            #  print "Update!!!"
            self.mu_update(state, action)
            self.sigma_update(state, action)


if __name__ == "__main__":
    import gym
    
    alpha = 0.1
    gamma = 0.99

    env = gym.make('FrozenLake-v0')

    test_agent = Agent(env, alpha, gamma)

    for i in xrange(100):
        state = env.reset()
        for j in xrange(100):
            env.render()
            #  action = env.action_space.sample()
            action = test_agent.actor(state)
            print "action : ", action

            next_state, reward, episode_end, info = env.step(action)

            test_agent.train(state, action, next_state, reward, episode_end)
            
            state = next_state

            if episode_end:
                break
        
        print "state : ", state, ", action : ", action, ", next_state : ", next_state, ", reward : ", reward, ", episode_end : ", episode_end

