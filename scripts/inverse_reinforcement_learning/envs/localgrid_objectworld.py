#!/usr/bin/env python
#coding:utf-8

import numpy as np

from objectworld import Objectworld

class LocalgridObjectworld(Objectworld):
    def __init__(self, g_rows, g_cols, R_max, noise, n_objects, seed, l_rows, l_cols):
        self.ow = Objectworld(g_rows, g_cols, R_max, noise, n_objects, seed)

        self.l_rows = l_rows
        self.l_cols = l_cols
        self.local_grid = np.zeros([self.l_rows, self.l_cols])

    def extract_local_grid(self, state):
        local_grid = self.local_grid
        #  print "state : ", state
        y, x = state
        center_y = int(self.l_rows / 2)
        center_x = int(self.l_cols / 2)
        #  print "center_y : ", center_y
        #  print "center_x : ", center_x
        

        for l_y in xrange(self.l_rows):
            for l_x in xrange(self.l_cols):
                g_y = y + (l_y - center_y)
                g_x = x + (l_x - center_x)
                #  print "g_y : ", g_y
                #  print "g_x : ", g_x 
                if g_x < 0 or g_y < 0:
                    local_grid[l_y, l_x] = 0
                else:
                    local_grid[l_y, l_x] = self.ow.grid[g_y, g_x]
        
        return local_grid
    
    def get_sample_action(self):
        return self.ow.get_action_sample()

    def reset(self, start_position=[0,0]):
        self.ow.reset()
        self.local_grid = self.extract_local_grid(self.ow.state_)
        return [self.ow.state_, self.local_grid]

    def step(self, actioni, reward_map=None):
        next_state, reward, done, info = self.ow.step(action, reward_map)
        self.local_grid = self.extract_local_grid(next_state)
        return [next_state, self.local_grid], reward, done, info



if __name__ == "__main__":
    rows = 15
    cols = 15
    R_max = 1.0
    noise = 0.0
    n_objects = 50
    seed = 1

    l_rows = 3
    l_cols = 3

    lg_ow = LocalgridObjectworld(rows, cols, R_max, noise, n_objects, seed, l_rows, l_cols)
    print "global_grid : "
    print lg_ow.ow.grid

    print "local_grid"
    print lg_ow.local_grid

    reward_map = lg_ow.ow.grid.transpose().reshape(-1)
    print "reward_map : "
    print reward_map
    
    max_episode = 100
    max_step = 50
    for i in xrange(max_episode):
        print "================================"
        print "episode : ", i
        observation = lg_ow.reset()

        for j in xrange(max_step):
            print "-------------------------------"
            print "step : ", j

            print "state : ", observation[0]
            print "local map : "
            print observation[1]
        
            action = lg_ow.get_sample_action()
            print "action : ", action

            observation, reward, done, info = lg_ow.step(action, reward_map)
            print "state : ", observation[0]
            print "local_map : "
            print observation[1]

            print "reward : ", reward
            print "episode_end : ", done
            if done:
                break
