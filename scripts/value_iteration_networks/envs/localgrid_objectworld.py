#!/usr/bin/env python
#coding:utf-8

import numpy as np
np.set_printoptions(suppress=True, threshold=np.inf)

import copy
import math

from object_world import Objectworld
import sys
sys.path.append('../')
from agents.a_star_agent import AstarAgent

class LocalgridObjectworld(Objectworld):
    def __init__(self, g_rows, g_cols, g_goal, R_max, noise, n_objects, seed, mode, l_rows, l_cols, l_goal_range):
        self.ow = Objectworld(g_rows, g_cols, g_goal, R_max, noise, n_objects, seed, mode=mode)

        self.l_rows = l_rows
        self.l_cols = l_cols
        self.l_n_state = self.l_rows * self.l_cols
        self.l_goal_range = l_goal_range

        self.local_grid = np.empty([self.l_rows, self.l_cols])
        self.local_grid.fill(-1)
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
        #  self.get_local_goal()
        self.get_local_goal_a_star(state)
        local_grid = self.extract_local_grid(state)
        local_grid_ = copy.deepcopy(local_grid)
        #  print "self.local_goal : ", self.local_goal
        local_grid_[self.local_goal] = 1
        return local_grid, local_grid_

    def extract_local_grid(self, state):
        self.local_grid = np.empty([self.l_rows, self.l_cols])
        self.local_grid.fill(-1)
        local_grid = self.local_grid
        #  print "state : ", state
        y, x = state
        center_y = int(self.l_rows / 2)
        center_x = int(self.l_cols / 2)
        #  print "center_y : ", center_y
        #  print "center_x : ", center_x
        min_y = int(y - center_y)
        max_y = int(y + center_y) + 1
        min_x = int(x - center_x)
        max_x = int(x + center_x) + 1
        if min_y < 0:
            min_y = 0
        if self.ow.rows < max_y:
            max_y =  self.ow.rows
        if min_x < 0:
            min_x = 0
        if self.ow.cols < max_x:
            max_x = self.ow.cols

        diff_min_y = min_y - y
        diff_max_y = max_y - y
        diff_min_x = min_x - x
        diff_max_x = max_x - x

        local_min_y = int(center_y + diff_min_y)
        local_max_y = int(center_y + diff_max_y)
        local_min_x = int(center_x + diff_min_x)
        local_max_x = int(center_x + diff_max_x)
        
        local_grid[local_min_y:local_max_y, local_min_x:local_max_x] = \
                self.ow.grid[min_y:max_y, min_x:max_x]
        
        
        return local_grid

    def get_local_goal(self):
        g_dist = math.sqrt((self.ow.goal[0]-self.ow.state_[0])**2 + (self.ow.goal[1]-self.ow.state_[1])**2)
        #  print "g_dist : ", g_dist
        #  print math.degrees(math.atan2((self.ow.goal[0]-self.ow.state_[0]), (self.ow.goal[1]-self.ow.state_[1])))
        theta = math.atan2((self.ow.goal[0]-self.ow.state_[0]), (self.ow.goal[1]-self.ow.state_[1]))

        diff_theta = (math.pi/2.0) / self.l_goal_range[0]
        #  print math.degrees(diff_theta)
        i = theta / diff_theta
        #  print i

        diff_y = int(self.l_goal_range[1] / 2)
        diff_x = int(self.l_goal_range[0] / 2)

        center_y = int(self.l_rows / 2)
        center_x = int(self.l_cols / 2)
        #  print "center_y : ", center_y
        #  print "center_x : ", center_x

        end_y = center_y + diff_y
        end_x = center_x + diff_x
        #  print "end_y : ", end_y
        #  print "end_x : ", end_x

        l_dist = math.sqrt((end_y-center_y)**2 + ((end_x-center_x)**2))
        #  print "l_dist : ", l_dist
        l_goal_y = (self.ow.goal[0] - self.ow.state_[0]) + center_y
        l_goal_x = (self.ow.goal[1] - self.ow.state_[1]) + center_x
        if g_dist < l_dist and (l_goal_y < end_y and l_goal_x < end_x):
            print "l_goal_y, l_goal_x : ", l_goal_y, l_goal_x
            self.local_goal = (l_goal_y, l_goal_x)
            #  print "self.local_goal(g_dist < l_dist) : ", self.local_goal
        else:
            local_goal_candidate = [(yi, end_x) for yi in xrange(center_y, end_y)]
            local_goal_candidate.append((end_y, end_x))
            local_goal_candidate.extend([(end_y, xi) for xi in xrange(center_x, end_x)][::-1])
            #  print local_goal_candidate


            if i == self.l_goal_range[0]:
                self.local_goal = local_goal_candidate[self.l_goal_range[0]-1]
                #  print local_goal_candidate[self.l_rows-1]
            else:
                self.local_goal = local_goal_candidate[int(i)]
                #  print local_goal_candidate[int(i)]

    def get_local_goal_a_star(self, state):
        a_agent = AstarAgent(self.ow)
        a_agent.get_shortest_path(state)
        if a_agent.found:
            pass
            #  print "a_agent.state_list : "
            #  print a_agent.state_list
            #  print "a_agent.shrotest_action_list : "
            #  print a_agent.shortest_action_list
            #  env.show_policy(a_agent.policy.transpose().reshape(-1))
            path_data = a_agent.show_path()
            print "view_path : "
            a_agent.view_path(path_data['vis_path'])
        
        center_y = int(self.l_rows / 2)
        center_x = int(self.l_cols / 2)
        near_state_list = [list(np.asarray(path_state) - np.asarray([state[0], state[1]]))\
                for path_state in a_agent.state_list \
                if (path_state[0]-state[0]) <= (self.l_rows-1)/2 \
                and (path_state[1]-state[1]) <= (self.l_cols-1)/2]
        #  print "near_state_list : ", near_state_list
        distance_list = [near_state[0]**2 + near_state[1]**2 \
                for near_state in near_state_list]
        #  print "distance_list", distance_list
        #  print "self.l_goal_range : ", self.l_goal_range 
        reff_distance = (self.l_goal_range[0]-1-center_y)**2 + (self.l_goal_range[1]-1-center_x)**2
        #  print "reff_distance : ", reff_distance
        index = np.abs(np.asarray(distance_list) - reff_distance).argmin()
        #  print "index : ", index

        #  index = np.argmax(np.asarray(distance_list))
        #  print "index : ", index
        self.local_goal = tuple(np.asarray(near_state_list[index]) + np.asarray([center_y, center_x]))
        #  print "self.local_goal : ", local_goal



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
    
    def show_global_objectworld(self):
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
    g_goal = [rows-1, cols-1]
    R_max = 10.0
    noise = 0.0
    n_objects = 100
    seed = 1

    mode = 1

    l_rows = l_cols = 5
    l_goal_range = [l_rows, l_cols] 

    lg_ow = LocalgridObjectworld(rows, cols, g_goal, R_max, noise, n_objects, seed, mode, l_rows, l_cols, l_goal_range)
    #  print "global_grid : "
    #  print lg_ow.ow.grid
    lg_ow.show_global_objectworld()

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
            print "action : ", action, " (", lg_ow.ow.dirs[action], ")"

            observation, reward, done, info = lg_ow.step(action, reward_map)
            print "state : ", observation[0]
            print "local_map : "
            print observation[1]
            #  lg_ow.show_global_objectworld()

            print "reward : ", reward
            print "episode_end : ", done
            if done:
                break
