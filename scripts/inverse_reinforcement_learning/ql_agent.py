#!/usr/bin/env python
#coding:utf-8

import numpy as np

class Agent:
    def __init__(self, state_num, action_num, gamma, alpha):
        self.state_num = state_num
        self.action_num = action_num
        
        self.alpha = alpha
        self.gamma = gamma

        self.q_table = np.zeros((self.state_num ,self.action_num), dtype=np.float32)
        self.epsilon = 0.1

    def get_action(self, state, evaluation=False):
        if not evaluation:
            if np.random.rand() < self.epsilon:
                #  print "Random action!!"
                return int(np.round(np.random.rand() * (self.action_num-1)))
            else:
                #  print "Greedy action!!"
                max_index_list = \
                        np.array(np.where(self.q_table[state] == self.q_table[state].max()))
                if len(max_index_list[0]) > 1:
                    np.random.shuffle(max_index_list[0])
                    return max_index_list[0][0]
                else:
                    return np.argmax(self.q_table[state])
        else:
            return np.argmax(self.q_table[state])

    def q_update(self, state, next_state, action, reward):
        self.q_table[state][action] = \
                (1-self.alpha)*self.q_table[state][action] \
                + self.alpha*(reward+self.gamma*np.max(self.q_table[next_state]))


if __name__ == "__main__":
    def state2index(state, rows, cols):
        return state[0] * cols + state[1]
    
    rows = 7
    cols = 7
    maze = np.zeros((rows, cols), dtype=np.int32)
    state_num = len(maze) * len(maze[0])
    print "state_num : ", state_num
    action_num = 5
    gamma = 0.5
    alpha = 0.3
    test_agent = Agent(state_num, action_num, gamma, alpha)
    print "test_agent.q_table : "
    print test_agent.q_table

    state = [1, 3]
    index = state2index(state, rows, cols)
    print "index : "
    print index
    action = test_agent.get_action(index)
    print "action : "
    print action
    
    next_state = [1, 4]
    next_index = state2index(next_state, rows, cols)
    print "next_index : "
    print next_index
    reward = 10
    for i in xrange(10):
        test_agent.q_update(index, next_index, action, reward)
        print "test_agent.q_table : "
        print test_agent.q_table
