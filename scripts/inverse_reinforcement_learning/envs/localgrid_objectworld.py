#!/usr/bin/env python
#coding:utf-8

import numpy as np
np.set_printoptions(suppress=True, threshold=np.inf)

import copy
import math

from objectworld import Objectworld

class LocalgridObjectworld(Objectworld):
    def __init__(self, g_rows, g_cols, R_max, noise, n_objects, seed, l_rows, l_cols):
        self.ow = Objectworld(g_rows, g_cols, R_max, noise, n_objects, seed)

        self.l_rows = l_rows
        self.l_cols = l_cols
        self.l_n_state = self.l_rows * self.l_cols
        self.local_grid = np.zeros([self.l_rows, self.l_cols])
        self.local_goal = None

    def local_state2index(self, state):
        # state[0] : y
        # state[1] : x
        return state[0] + self.l_cols * state[1]

    def local_index2state(self, index):
        state = [0, 0]
        state[0] = index % self.l_cols    # y
        state[1] = index / self.l_cols    # x
        return state

    def create_local_grid(self, state):
        self.get_local_goal()
        local_grid = self.extract_local_grid(state)
        local_grid_ = copy.deepcopy(local_grid)
        local_grid_[self.local_goal] = 1
        return local_grid, local_grid_

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
                if (g_x < 0 or self.ow.cols-1 < g_x) or (g_y < 0 or self.ow.rows-1 < g_y):
                    local_grid[l_y, l_x] = -1
                else:
                    local_grid[l_y, l_x] = self.ow.grid[g_y, g_x]
        
        return local_grid

    def get_local_goal(self):
        g_dist = math.sqrt((self.ow.goal[0]-self.ow.state_[0])**2 + (self.ow.goal[1]-self.ow.state_[1])**2)
        #  print "g_dist : ", g_dist
        #  print math.degrees(math.atan2((self.ow.goal[0]-self.ow.state_[0]), (self.ow.goal[1]-self.ow.state_[1])))
        theta = math.atan2((self.ow.goal[0]-self.ow.state_[0]), (self.ow.goal[1]-self.ow.state_[1]))

        diff_theta = (math.pi/2.0) / self.l_rows
        #  print math.degrees(diff_theta)
        i = theta / diff_theta
        #  print i

        center_y = int(self.l_rows / 2)
        center_x = int(self.l_cols / 2)

        l_dist = math.sqrt(((self.l_rows-1)-center_y)**2 + ((self.l_cols-1)-center_x)**2)
        #  print "l_dist : ", l_dist
        if g_dist < l_dist:
            l_goal_y = self.ow.goal[0] - self.ow.state_[0]
            l_goal_x = self.ow.goal[1] - self.ow.state_[1]
            #  print "l_goal_y, l_goal_x : ", l_goal_y, l_goal_x
            self.local_goal = (l_goal_y+center_y, l_goal_x+center_x)
            #  print "self.local_goal(g_dist < l_dist) : ", self.local_goal
        else:
            local_goal_candidate = [(yi, self.l_cols-1) for yi in xrange(center_y, self.l_rows-1)]
            local_goal_candidate.append((self.l_rows-1, self.l_cols-1))
            local_goal_candidate.extend([(self.l_rows-1, xi) for xi in xrange(center_x, self.l_cols-1)][::-1])

            #  for xi in xrange(self.l_cols-2, center_x+1, -1):
                #  local_goal_candidate.append((self.l_rows-1, xi))
            #  print local_goal_candidate

            if i == self.l_rows:
                self.local_goal = local_goal_candidate[self.l_rows-1]
                #  print local_goal_candidate[self.l_rows-1]
            else:
                self.local_goal = local_goal_candidate[int(i)]
                #  print local_goal_candidate[int(i)]
        

    def get_sample_action(self):
        return self.ow.get_action_sample()

    def reset(self, start_position=[0,0]):
        self.ow.reset()
        self.get_local_goal()
        self.local_grid = self.extract_local_grid(self.ow.state_)
        return [self.ow.state_, self.local_grid]

    def step(self, action, reward_map=None):
        next_state, reward, done, info = self.ow.step(action, reward_map)
        self.local_grid, _ = self.create_local_grid(next_state)
        info["local_grid"] = _
        return [next_state, self.local_grid], reward, done, info
    
    def show_global_grid(self):
        global_grid = copy.deepcopy(self.ow.grid)
        if self.ow.state_ != None:
            global_grid[tuple(self.ow.state_)] = 2
        for row in global_grid:
            print "|",
            for i in row:
                print "%2d" % i,
            print "|"



if __name__ == "__main__":
    rows = cols = 50
    R_max = 10.0
    noise = 0.0
    n_objects = 1000
    seed = 1

    l_rows = l_cols = 5

    lg_ow = LocalgridObjectworld(rows, cols, R_max, noise, n_objects, seed, l_rows, l_cols)
    #  print "global_grid : "
    #  print lg_ow.ow.grid
    lg_ow.show_global_grid()

    print "local_grid"
    print lg_ow.local_grid

    reward_map = lg_ow.ow.grid.transpose().reshape(-1)
    print "reward_map : "
    print reward_map
    
    max_episode = 1
    max_step = 500
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
            lg_ow.show_global_grid()

            print "reward : ", reward
            print "episode_end : ", done
            if done:
                break
