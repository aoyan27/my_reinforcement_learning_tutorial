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

class Agent:
    def __init__(self, env, alpha, gamma):
        #  print "env.onbservation_space.n : ", env.observation_space.n
        #  print "env.action_space.n : ", env.action_space.n
        self.alpha = alpha
        self.gamma = gamma
        self.num_state = env.observation_space.n
        self.num_action = env.action_space.n
        self._q_table = np.zeros((self.num_state, self.num_action), dtype=np.float32)
        
        self.action_list = np.arange(self.num_action, dtype=np.int32)

        self.epsilon = 1.0

    def get_action(self, state, evoluation=False):
        if not evoluation:
            prob =  np.random.rand()
            if self.epsilon > prob:
                #  print "Random"
                return np.random.randint(0, self.num_action)
            else:
                #  print "Greedy"
                max_index_list = np.where(self._q_table[state] == self._q_table[state].max())
                if(len(max_index_list[0]) > 1):
                    np.random.shuffle(max_index_list[0])
                    return self.action_list[max_index_list[0][0]]
                else:
                    return self.action_list[self._q_table[state].argmax()]

        else:
            return self.action_list[self._q_table[state].argmax()]
    def epsilon_decay(self, count):
        if self.epsilon > 0.1:
            if count % 100000:
                self.epsilon -= 1e-6
        else:
            self.epsilon = 0.1


    def q_update(self, state, next_state, action, reward, episode_end):
        #  if not episode_end:
        self._q_table[state][action] = (1-self.alpha)*self._q_table[state][action] + self.alpha*(reward+self.gamma*np.max(self._q_table[next_state]))
        #  else:
            #  self._q_table[state][action] = reward

        #  if self._q_table[state][action] != 0:
            #  print self._q_table[state][action], "\t", reward
        #  else:
            #  print "(", self._q_table[state][action], ")\t", "(", reward, ")"

    def display_q_table(self):
        print self._q_table
    
    def save_model(self, path):
        print "Save Q_table!!"
        np.save(path, self._q_table)

    def load_model(self, path):
        print "Load Q_table!!"
        self._q_table = np.load(path)



if __name__ == "__main__":
    import gym
    env = gym.make('FrozenLake-v0')
    test_agent = Agent(env)

    state = 0
    action = test_agent.get_action(state)
    print action
    
    next_state = 1
    reward = 1
    test_agent.q_update(state, next_state, action, reward)
    test_agent.display_q_table()

