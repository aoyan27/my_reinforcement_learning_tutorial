#!/usr/bin/env python
#coding:utf-8
"""
迷路探索
   5x5の大きさの迷路を探索する
   状態 : 自身のマップ上の位置(Q_tableはマップ全体に確保される)
   行動 : 上下左右の４パターン(上 : a=0 左 : a=1 下 : a=2 右 : a=3)
   報酬 : ゴールに到達した時
"""

import numpy as np

class Agent:
    ALPHA = 0.5
    GAMMA = 0.9

    def __init__(self, maze, action_num):
        self.action_num = action_num
        self._maze = maze 
        self._q_table = np.zeros((len(self._maze)*len(self._maze[0]),self.action_num), dtype=np.float32)
        self.epsilon = 0.1
    
    def state2index(self, state):
        return state[0] * len(self._maze[0]) + state[1]

    def epsilon_greedy(self, state, evaluation=False):
        if not evaluation:
            if np.random.rand() < self.epsilon:
                #  print "Random action!!"
                return int(np.round(np.random.rand() * (self.action_num-1)))
            else:
                index = self.state2index(state)
                #  print "Greedy action!!"
                max_index_list = np.array(np.where(self._q_table[index] == self._q_table[index].max()))
                if len(max_index_list[0]) > 1:
                    np.random.shuffle(max_index_list[0])
                    return max_index_list[0][0]
                else:
                    return np.argmax(self._q_table[index])
        else:
            index = self.state2index(state)
            return np.argmax(self._q_table[index])

    def q_update(self, state, next_state, action, reward):
        index = self.state2index(state)
        next_index = self.state2index(next_state)
        self._q_table[index][action] = (1-self.ALPHA)*self._q_table[index][action] + self.ALPHA*(reward+self.GAMMA*np.max(self._q_table[next_index]))

    def display_q_table(self):
        print self._q_table

    def show_policy(self):
        action_display_list = ['^', '<', 'v', '>']
        for i in xrange(len(self._maze)):
            for j in xrange(len(self._maze[0])):
                state = [i, j]
                if self._maze[i][j] == 0:
                    if np.max(self._q_table[self.state2index(state)])  != 0:
                        action = self.epsilon_greedy(state, evaluation=True)
                        print '%2s' % action_display_list[action],
                    else:
                        print '%2d' % self._maze[i][j],
                else:
                    print '%2d' % self._maze[i][j],
            print 

if __name__ == "__main__":
    maze = np.zeros((7,7), dtype=np.int32)
    action_num = 4
    test_agent = Agent(maze, action_num)
    test_agent.display_q_table()

    state = [1, 3]
    action = test_agent.epsilon_greedy(state)
    print action
    
    next_state = [1, 4]
    reward = 10
    for i in xrange(10):
        test_agent.q_update(state, next_state, action, reward)
        test_agent.display_q_table()
    test_agent.show_policy()
