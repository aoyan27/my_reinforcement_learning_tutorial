#!/usr/bin/env python
#coding:utf-8

import numpy as np
np.set_printoptions(suppress=True, threshold=np.inf)


class ImmediateRewardAgent:
    def __init__(self, env):
        self.env = env
        self.n_action = env.ow.n_action
        self.actions = [a for a in xrange(self.n_action)]
        self.rows = env.l_rows
        self.cols = env.l_cols

    def get_action(self, reward_map):
        center_y = int(self.rows / 2)
        center_x = int(self.cols / 2)
        #  print "center_y, center_x : ", center_y, center_x
        
        reward_list = np.array([])
        for a in xrange(self.n_action-1):
            next_state = self.move([center_y, center_x], a)
            reward = reward_map[self.env.local_state2index(next_state)]
            reward_list = np.append(reward_list, reward)
        reward_list[1] = 0.0
        reward_list[3] = 0.0
        print "reward_list", reward_list
        optimal_action = np.argmax(reward_list)
        #  print optimal_action
        return optimal_action
    


    def move(self, state, action):
        y, x = state
        next_y, next_x = state

        if action == 0:
            #  right
            next_x = x + 1
        elif action == 1:
            #  left
            next_x = x - 1
        elif action == 2:
            #  down
            next_y = y + 1
        elif action == 3:
            #  up
            next_y = y - 1
        else:
            #  stay
            next_x = x
            next_y = y

        return [next_y, next_x]
        
