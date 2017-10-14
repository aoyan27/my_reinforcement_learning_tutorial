#!/usr/bin/env python
#coding:utf-8

import numpy as np

class Gridworld:
    def __init__(self, rows, cols, goal, noise):
        self.rows = rows
        self.cols = cols

        self.noise = noise

        self.grid = np.zeros((self.rows, self.cols))
        # +------------> x
        # |
        # |
        # |
        # |
        # |
        # v
        # y
        self.n_state = self.rows*self.cols

        self.action_list = [0, 1, 2, 3, 4]
        self.n_action = len(self.action_list)
        self.dirs = {0: '>', 1: '<', 2: 'v', 3: '^', 4: '-'}
        
        self.goal = goal

        self._state = None
        self._out_of_range = False
        
    def state2index(self, state):
        #  state[1] : x
        #  state[0] : y
        return state[1] + self.cols*state[0]

    def index2state(self, index):
        state = [0, 0]
        state[1] = index % self.cols #  x
        state[0] = index / self.cols #  y
        return state

    def move(self, state, action):
        y, x = state
        if action == 0:
            #  right
            x = x + 1
        elif action == 1:
            #  left
            x = x - 1
        elif action == 2:
            #  down
            y = y - 1
        elif action == 3:
            #  up
            y = y + 1
        else:
            # stay
            x = x
            y = y

        if x < 0:
            x = 0
            self._out_of_range = True
        elif x > (self.cols - 1):
            x = self.cols - 1
            self._out_of_range = True

        if y < 0:
            y = 0
            self._out_of_range = True
        elif y < (self.rows - 1):
            y = self.rows - 1
            self._out_of_range = True

        return [x, y]

    def get_next_state_and_probs(self, state, action):
        transition_probability = 1 - self.noise
        probs = np.zeros([self.n_action])
        probs[action] = trasition_probability
        probs += self.noise / self.n_action
        
        next_state_list = []

        for a in xrange(self.n_action):
            if state != list(self.goal):
                next_state = self.move(state, a)
                next_state_list.append(next_state)

                if self._out_of_range:
                    probs[self.n_action-1] += probs[a]
                    probs[a] = 0
            else:
                next_state = state

                if a != self.n_action - 1:
                    probs[self.n_action-1] += probs[a]
                    probs[a] = 0
                next_state_list.append(next_state)

        return next_state_list, prob

  




if __name__=="__main__":
    rows = 5
    cols = 5
    noise = 0.0
    goal = [1, 2]
    gw = Gridworld(rows, cols, goal, noise)

    print gw.grid

    print gw.n_state
    print gw.n_action
    
    count = 0
    state_list = {}
    for i in xrange(gw.rows):
        for j in xrange(gw.cols):
            state_list[count] = [i, j]
            
            print "%2d " % count, 
            count += 1
        print ""

    print state_list

    for i in xrange(gw.n_state):
        print gw.index2state(i)
        print gw.state2index(state_list[i])
