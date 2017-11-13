#!/usr/bin/env python
#coding:utf-8

import numpy as np

class Objectworld:
    def __init__(self, rows, cols, R_max, noise, n_objects, seed):
        np.random.seed(seed)

        self.rows = rows
        self.cols = cols
        self.n_state = self.rows * self.cols
        
        self.R_max = R_max
        
        self.grid = np.zeros([self.rows, self.cols])
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
        
        self.n_objects = n_objects
        self.objects = self.set_objects()
        for i in xrange(len(self.objects)):
            self.grid[self.objects[i]] = -1
        print self.grid

        self.action_list = [0, 1, 2, 3, 4]
        self.n_action = len(self.action_list)
        self.dirs = {0: '>', 1: '<', 2:'v', 3: '^', 4: '-'}

        self.state_ = None

        self.out_of_range_ = None
        self.collision_ = None

    def set_objects(self):
        objects_ = []
        i = 0
        while i < self.n_objects:
            #  print " i : ", i
            y = np.random.randint(0, self.rows)
            x = np.random.randint(0, self.cols)
            i += 1
            if (y, x) == (0, 0) or (y, x) == self.goal:
                i -= 1
            else:
                objects_.append((y, x))
            #  print "(y, x) : ", (y, x)
        #  print "objects_ : ", objects_
        return objects_
    
    def state2index(self, state):
        # state[0] : y
        # state[1] : x
        return state[0] + self.cols * state[1]

    def index2state(self, index):
        state = [0, 0]
        state[0] = index % self.cols    # y
        state[1] = idnex / self.cols    # x
        return state

    def move(self, state, action):
        y, x = state
        next_x = 0
        next_y = 0
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
        
        out_of_range = False
        if next_y < 0 or (self.rows-1) < next_y:
            print "y, out_of_range!!!!"
            next_y = y
            out_of_range = True

        if next_x < 0 or (self.cols-1) < next_x:
            print "x, out of range!!!!!"
            next_x = x
            out_of_range = True

        collision = False
        if self.grid[next_y, next_x] == -1:
            print "collision!!!!!"
            collision = True
            if action == 0 or action == 1:
                next_x = x
            elif action == 2 or action == 3:
                next_y = y

        return [next_y, next_x], out_of_range, collision

    def get_next_state_and_probs(self, state, action):
        transition_probability = 1 - self.noise
        probs = np.zeros([self.n_action])
        probs[int(action)] = transition_probability
        probs += self.noise / self.n_action
        #  print "probs : "
        #  print probs
        next_state_list = []

        for a in xrange(self.n_action):
            if state != list(self.goal):
                #  print "state : ", state
                next_state, out_of_range, collision = self.move(state, a)
                self.out_of_range_ = out_of_range
                self.collision_ = collision
                #  print "next_state() : "
                #  print next_state
                next_state_list.append(next_state)

                if out_of_range:
                    probs[self.n_action-1] += probs[a]
                    probs[a] = 0
            else:
                next_state = state
                #  print "probs[", a, "] : ", probs[a]
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
        



if __name__ == "__main__":
    rows = 5
    cols = 5
    R_max = 1.0
    noise = 0.0
    n_objects = 5
    seed = 1

    ow = Objectworld(rows, cols, R_max, noise, n_objects, seed)
    
    state = [0, 0]
    action = 0
    for i in xrange(10):
        print "state : ", state
        print "action : ", action
        state, out_of_range, collision = ow.move(state, action)
        print "next_state : ", state

