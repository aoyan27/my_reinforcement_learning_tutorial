#!/usr/bin.env python
#coding:utf-8

import numpy as np

class Gridworld:
    def __init__(self, rows, cols, R_max):
        self.rows = rows
        self.cols = cols
        self.n_state = self.rows * self.cols

        self.R_max = R_max

        self.grid = np.zeros((self.rows, self.cols))
        # +----------------> x
        # |
        # |
        # |
        # |
        # |
        # |
        # V
        # y
        self.goal = (self.rows-1, self.cols-1)
        self.grid[self.goal] = self.R_max

        self.action_list = [0, 1, 2, 3, 4]
        self.n_action = len(self.action_list)
        self.dirs = {0: '>', 1: '<', 2: 'v', 3: '^', 4: '-'}

    def state2index(self, state):
        #  state[0] : x
        #  state[1] : y
        return state[1] + self.cols * state[0]

    def index2state(self, index):
        state = [0, 0]
        state[1] = index % self.cols
        state[0] = index / self.cols
        return state

    def get_next_state_and_probs(self, state, action, noise):
        transition_probability = 1 - noise
        probs = np.zeros([self.n_action])
        probs[action] = transition_probability
        probs += noise / self.n_action
        print "probs : "
        print probs
        next_state_list = []

        for a in xrange(self.n_action):
            if state != list(self.goal):
                x, y = state
                #  print "x : ", x
                #  print "y : ", y
                if a == 0:
                    #  right
                    x = x + 1
                elif a == 1:
                    #  left
                    x = x - 1
                elif a == 2:
                    #  down
                    y = y + 1
                elif a == 3:
                    #  up
                    y = y - 1
                else:
                    #  stay
                    x = x
                    y = y
                 
                out_of_range = False
                if x < 0:
                    x = 0
                    out_of_range = True
                elif x > (self.cols-1):
                    x = self.cols - 1
                    out_of_range = True

                if y < 0:
                    y = 0
                    out_of_range = True
                elif y > (self.rows-1):
                    y = self.rows - 1
                    out_of_range = True

                next_state = [x, y]
                #  print "next_state() : "
                #  print next_state
                next_state_list.append(next_state)

                if out_of_range:
                    probs[self.n_action-1] += probs[a]
                    probs[a] = 0

            else:
                next_state = state
                print "probs[", a, "] : ", probs[a]
                if a != self.n_action-1:
                    probs[self.n_action-1] += probs[a]
                    probs[a] = 0

                next_state_list.append(next_state)
                #  print "next_state_ : "
                #  print next_state
        #  print "next_state_list : "
        #  print next_state_list
            
        #  print "probs_ : "
        #  print probs


        return next_state_list, probs
    
    def show_policy(self, policy):
        vis_policy = np.array([])
        for i in xrange(len(policy)):
            vis_policy = np.append(vis_policy, self.dirs[policy[i]])
            #  print self.dirs[policy[i]]
        print vis_policy.reshape((self.rows, self.cols))

