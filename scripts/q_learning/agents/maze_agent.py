#!/usr/bin/env python
#coding: utf-8
"""
迷路探索
   5x5の大きさの迷路を探索する
   状態 : 迷路全体　or 自身のxy座標
   行動 : 上下左右の４パターン(上 : a=0 左 : a=1 下 : a=2 右 : a=3)
   報酬 : ゴールに到達した時
"""

import numpy as np

class Agent:
    ALPHA = 0.5
    GAMMA = 0.9

    def __init__(self, maze, action_num):
        self.__action_num = action_num
        self.__maze = maze 
        self.__q_table = np.zeros((len(self.__maze)*len(self.__maze[0]),self.__action_num), dtype=np.float32)
        self.__epsilon = 0.1
        self.__action = np.arange(0, self.__action_num, dtype=np.int32)
    
    def state2index(self, state):
        return state[0] * len(self.__maze[0]) + state[1]

    def epsilon_greedy(self, state, evoluation=False):
        if not evoluation:
            if np.random.rand() < self.__epsilon:
                #  print "Random action!!"
                return int(np.round(np.random.rand() * (self.__action_num-1)))
            else:
                index = self.state2index(state)
                #  print "Greedy action!!"
                max_index_list = np.array(np.where(self.__q_table[index] == self.__q_table[index].max()))
                if len(max_index_list[0]) > 1:
                    np.random.shuffle(max_index_list[0])
                    return max_index_list[0][0]
                else:
                    return np.argmax(self.__q_table[index])
        else:
            index = self.state2index(state)
            return np.argmax(self.__q_table[index])

    def q_update(self, state, next_state, action, reward):
        index = self.state2index(state)
        next_index = self.state2index(next_state)
        self.__q_table[index][action] = (1-self.ALPHA)*self.__q_table[index][action] + self.ALPHA*(reward+self.GAMMA*np.max(self.__q_table[next_index]))

    def display_q_table(self):
        print self.__q_table

    def show_policy(self):
        action_display_list = ['^', '<', 'v', '>']
        for i in xrange(len(self.__maze)):
            for j in xrange(len(self.__maze[0])):
                state = [i, j]
                if self.__maze[i][j] == 0:
                    if np.max(self.__q_table[self.state2index(state)])  != 0:
                        action = self.epsilon_greedy(state, evoluation=True)
                        print '%2s' % action_display_list[action],
                    else:
                        print '%2d' % self.__maze[i][j],
                else:
                    print '%2d' % self.__maze[i][j],
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
