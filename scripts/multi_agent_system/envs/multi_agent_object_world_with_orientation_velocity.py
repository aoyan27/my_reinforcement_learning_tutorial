#!/usr/bin/env python
#coding:utf-8

import numpy as np
import sys
import copy
import math

class Objectworld:
    def __init__(self, rows, cols, n_objects, num_agent, noise,\
            start={0: [0, 0], 1: [4, 4]}, goal={0: [4, 4], 1: [0, 0]}, \
            orientation={0: 0.0, 1: 0.0}, \
            object_list=None, random_objects=True, seed=0, mode=0):
        np.random.seed(seed)

        self.mode = mode # mode=0 : 行動４パターン, mode=1 : 行動８パターン

        self.rows = rows
        self.cols = cols

        self.n_objects = n_objects

        self.object_list = object_list
        self.random_objects = random_objects

        self.num_agent = num_agent

        self.noise = noise

        self.R_max = 1.0

        self.grid = np.zeros((self.rows, self.cols))
        self.agent_grid = {0: np.zeros((self.rows, self.cols)), 1: np.zeros((self.rows, self.cols))}
        self.another_agent_position_with_grid = {0: np.zeros((self.rows, self.cols)), 1: np.zeros((self.rows, self.cols))}
        # +------------> x
        # |
        # |
        # |
        # |
        # |
        # v
        # y
        self.n_state = self.rows*self.cols

        self.action_list = None
        self.n_action = 0
        self.dirs = {}
        self.velocity_vector = {}
        self.set_action()

        self.relative_velocity_vector = {}
        self.set_relative_velocity_vector()

        self.start = {}
        self.start_index = None
        self.set_start(start)
        
        self.goal_index = None
        self.goal = {}
        self.set_goal(goal)


        self.orientation_res = None
        if self.mode == 0:
            self.orientation_res = 2.0*math.pi / 4.0
        elif self.mode == 1:
            self.orientation_res = 2.0*math.pi / 8.0

        self._state = {}
        self.set_state(self.start)
        #  print "self._state : ", self._state
        self._orientation = {}
        self.set_orientation(orientation)
        self.relative_orientation = {}
        self.get_relative_orientation()
        #  print "self.relative_orientation : ", self.relative_orientation


        self.objects = []
        self.set_objects()
        
        self.collisions_ = []

        self.agent_collision = False
       
        self.episode_end_states = {0: 'continue', 1: 'Success', 2: 'Faild'}

    def set_action(self):
        if self.mode == 0:    # mode=0 : 行動4パターン
            self.action_list = [0, 1, 2, 3, 4]
            self.n_action = len(self.action_list)
            self.dirs = {0: '>', 1: 'v', 2: '<', 3: '^', 4: '-'}
            self.velocity_vector = {0: [0, 1], 1: [1, 0], 2: [0, -1], 3: [-1, 0], 4: [0, 0]}
        elif self.mode == 1:    # mode=1 : 行動8パターン
            self.action_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            self.n_action = len(self.action_list)
            self.dirs = \
                    {0: '>', 1: 'dr', 2: 'v', 3: 'dl', 4: '<', 5: 'ul', 6: '^', 7: 'ur', 8: '-'}
            self.velocity_vector = \
                    {0: [0, 1], 1: [1, 1], 2: [1, 0], 3: [1, -1], 4: [0, -1], \
                    5: [-1, -1], 6: [-1, 0], 7: [-1, 1], 8: [0, 0]}


    
    def get_action_list_by_direction(self, state, agent_id):
        #  print "agent_id : ", agent_id
        #  print "state : ", state
        #  print "self.goal[agent_id] : ", self.goal[agent_id]
        relative_orientation = math.atan2(self.goal[agent_id][0]-state[0], \
                self.goal[agent_id][1]-state[1])
        #  print "relative_orientation : ", math.degrees(relative_orientation)
        if relative_orientation < 0.0:
            relative_orientation = 2.0*math.pi + relative_orientation
        #  print "relative_orientation : ", math.degrees(relative_orientation)
        dir_index = (relative_orientation+0.5*self.orientation_res)/self.orientation_res
        #  print "dir_index : ", dir_index
        #  print "len(self.action_list) : ", len(self.action_list)-1
        action_list_ = range(len(self.action_list)-1)
        #  print "action_list_ : ", action_list_
        if dir_index >= len(action_list_):
            dir_index = 0
        dir_index = int(dir_index)
        #  print "dir_index : ", dir_index

        action_list_by_direction = \
                [action_list_[0] if dir_index+i>=len(action_list_) \
                else action_list_[dir_index+i] for i in xrange(-1, 2)]
        #  print action_list_by_direction
        return action_list_by_direction

    def get_action_list_by_my_orientation(self, orientation):
        #  print "self.goal : ", self.goal
        #  print "orientation : ", math.degrees(orientation)
        if orientation < 0.0:
            orientation = 2.0*math.pi + orientation
        dir_index = (orientation+0.5*self.orientation_res)/self.orientation_res
        #  print "dir_index : ", dir_index
        #  print "len(self.action_list) : ", len(self.action_list)-1
        action_list_ = range(len(self.action_list)-1)
        #  print "action_list_ : ", action_list_
        if dir_index >= len(action_list_):
            dir_index = 0
        dir_index = int(dir_index)
        #  print "dir_index : ", dir_index

        action_list_by_my_orientation = \
                [action_list_[0] if dir_index+i>=len(action_list_) \
                else action_list_[dir_index+i] for i in xrange(-1, 2)]
        #  print action_list_by_my_orientation
        return action_list_by_my_orientation


    def set_agent_grid(self):
        self.agent_grid = {0: copy.deepcopy(self.grid), 1: copy.deepcopy(self.grid)}
        for i in xrange(self.num_agent):
            for j in xrange(self.num_agent):
                if i != j:
                    self.agent_grid[i][tuple(self._state[j])] = -1

    def set_another_agent_position_with_grid(self):
        self.another_agent_position_with_grid = \
                {0: np.zeros((self.rows, self.cols)), 1: np.zeros((self.rows, self.cols))}
        for i in xrange(self.num_agent):
            for j in xrange(self.num_agent):
                if i != j:
                    self.another_agent_position_with_grid[i][tuple(self._state[j])] = -1
    
    def set_orientation(self, orientation):
        self._orientation = orientation

    def set_orientation_random(self, orientation_list=None):
        if orientation_list is None:
            for agent_id in xrange(self.num_agent):
                self._orientation[agent_id] = np.random.rand() * 2.0*math.pi - math.pi
                print "self._orinetation : ", math.degrees(self._orientation[agent_id])
            print "self._orinetation : ", self._orientation
        else:
            for agent_id in xrange(self.num_agent):
                self._orientation[agent_id] = math.radians(np.random.choice(orientation_list, 1))
                #  print "self._orinetation : ", math.degrees(self._orientation[agent_id])
            #  print "self._orinetation : ", self._orientation

    def get_relative_orientation(self):
        for agent_id in xrange(self.num_agent):
            relative_orientation_ = self._orientation[agent_id] - \
                    math.atan2(self.goal[agent_id][0]-self._state[agent_id][0], \
                    self.goal[agent_id][1]-self._state[agent_id][1])
            if math.fabs(relative_orientation_) >= math.pi:
                if relative_orientation_ < 0.0:
                    relative_orientation_ += 2.0*math.pi
                elif relative_orientation_ > 0.0:
                    relative_orientation_ -= 2.0*math.pi
            self.relative_orientation[agent_id] = relative_orientation_

    def set_relative_velocity_vector(self, input_vector=None):
        if input_vector is None:
            stay_action = len(self.action_list) - 1 
            for agent_id in xrange(self.num_agent):
                self.relative_velocity_vector[agent_id] = self.velocity_vector[stay_action]
        else:
            self.relative_velocity_vector = input_vector

    def get_relative_velocity_vector(self, action):
        relative_velocity_vector_ = {}
        for agent_id in xrange(self.num_agent):
            relative_velocity_vector_[agent_id] = \
                    copy.deepcopy(self.velocity_vector[action[agent_id]])
        #  print "relative_velocity_vector_ : ", relative_velocity_vector_
        for agent_id in xrange(self.num_agent):
            for other_agent_id in xrange(self.num_agent):
                if agent_id != other_agent_id:
                    relative_velocity_vector_[agent_id][1] = \
                            self.velocity_vector[action[other_agent_id]][0]*\
                            math.sin(self._orientation[agent_id]) + \
                            self.velocity_vector[action[other_agent_id]][1]*\
                            math.cos(self._orientation[agent_id])

                    relative_velocity_vector_[agent_id][0] = \
                            self.velocity_vector[action[other_agent_id]][0]*\
                            math.cos(self._orientation[agent_id]) -\
                            self.velocity_vector[action[other_agent_id]][1]*\
                            math.sin(self._orientation[agent_id])

        #  print "other_agent_relative_velocity_vector_ : ", relative_velocity_vector_
        self.set_relative_velocity_vector(input_vector=relative_velocity_vector_)
        

    
    def set_state(self, state):
        self._state = state

    def set_start(self, start):
        self.start = start
        self._state = start
        self.set_agent_grid()
        self.set_another_agent_position_with_grid()

    def set_start_random(self, check_goal=False):
        if not check_goal:
            self.start_index = \
                    np.random.choice(xrange(self.n_state), self.num_agent, replace=False)
            #  print "self.start_index : ", self.start_index
            for i in xrange(self.num_agent):
                self.start[i] = self.index2state(self.start_index[i])
            self._state = self.start
            #  print "self.start", self.start
            self.set_agent_grid()
            self.set_another_agent_position_with_grid()
        else:
            while 1:
                ok_flag_list = []
                self.start_index = \
                        np.random.choice(xrange(self.n_state), self.num_agent, replace=False)
                for i in xrange(self.num_agent):
                    if self.grid[tuple(self.index2state(self.start_index[i]))] != -1:
                        ok_flag_list.append(True)
                        for j in xrange(self.num_agent):
                            if self.start_index[i] != self.goal_index[j]:
                                ok_flag_list.append(True)
                #  print "ok_flag_list : ", ok_flag_list
                if len(ok_flag_list) == self.num_agent*(1+self.num_agent):
                    break

            for i in xrange(self.num_agent):
                self.start[i] = self.index2state(self.start_index[i])
            self._state = self.start
            #  print "self.start", self.start
            self.set_agent_grid()
            self.set_another_agent_position_with_grid()

    def set_goal(self, goal):
        self.goal = goal

    def set_goal_random(self, check_start=True):
        if check_start:
            while 1:
                ok_flag_list = []
                self.goal_index = \
                        np.random.choice(xrange(self.n_state), self.num_agent, replace=False)
                for i in xrange(self.num_agent):
                    if self.grid[tuple(self.index2state(self.goal_index[i]))] != -1:
                        ok_flag_list.append(True)
                        for j in xrange(self.num_agent):
                            if self.goal_index[i] != self.start_index[j]:
                                ok_flag_list.append(True)
                #  print "ok_flag_list : ", ok_flag_list
                if len(ok_flag_list) == self.num_agent*(1+self.num_agent):
                    break

            for i in xrange(self.num_agent):
                self.goal[i] = self.index2state(self.goal_index[i])
            #  print "self.goal", self.goal
        else:
            self.goal_index = \
                    np.random.choice(xrange(self.n_state), self.num_agent, replace=False)
            #  print "self.goal_index : ", self.goal_index
            for i in xrange(self.num_agent):
                self.goal[i] = self.index2state(self.goal_index[i])
            #  print "self.goal", self.goal

    def set_objects(self):
        self.objects = []
        self.grid = np.zeros([self.rows, self.cols])
        #  self.set_goal(self.goal)
        n_objects_ = np.random.randint(0, self.n_objects)
        #  print "n_objects_ : ", n_objects_
        if self.random_objects:
            i = 0
            while i <= n_objects_:
                #  print " i : ", i
                y = np.random.randint(0, self.rows)
                x = np.random.randint(0, self.cols)
                #  print "self.start : ", self.start
                #  print "self.goal : ", self.goal
                #  print "(y, x)_ : ", (y, x)
                check_list = []
                for j in xrange(self.num_agent):
                    check_list.append(self.start[j])
                    check_list.append(self.goal[j])
                #  print "check_list : ", check_list
                matched_list = []
                for check in check_list:
                    if (y, x) == tuple(check):
                        matched_list.append((y, x))
                
                if len(matched_list) == 0:
                    self.objects.append((y, x))
                    self.grid[y, x] = -1
                    #  print "(y, x) : ", (y, x)
                    i += 1
        else:
            self.objects = self.object_list

        self.set_agent_grid()
        self.set_another_agent_position_with_grid()
        #  print self.objects


    def state2index(self, state):
        #  state[1] : x
        #  state[0] : y
        return state[1] + self.cols*state[0]

    def index2state(self, index):
        state = [0, 0]
        state[1] = index % self.cols #  x
        state[0] = index / self.cols #  y
        return state

    def get_sample_action_single_agent(self):
        action = None
        if self.mode == 0:
            action = np.random.randint(0, 4)
        elif self.mode == 1:
            action = np.random.randint(0, 8)
        else:
            sys.stderr.write('Error occurred! Please set mode 0 or 1 !!')
            sys.exit()

        return action

    def get_sample_action(self):
        action = {}
        if self.mode == 0:
            for i in xrange(self.num_agent):
                action[i] = np.random.randint(0, 4)
        elif self.mode == 1:
            for i in xrange(self.num_agent):
                action[i] = np.random.randint(0, 8)
        else:
            sys.stderr.write('Error occurred! Please set mode 0 or 1 !!')
            sys.exit()

        return action

    def move(self, state, action, grid_range=None, grid=None, reflect=1):
        #  print "action : ", action
        if grid_range is None:
            grid_range = [self.rows, self.cols]
        if grid is None:
            grid = self.grid

        y, x = state
        next_y, next_x = state

        next_y = y + reflect*self.velocity_vector[action][0]
        next_x = x + reflect*self.velocity_vector[action][1]
        #  print "next_y : ", next_y
        #  print "next_x : ", next_x
        

        out_of_range = False
        if next_y < 0 or (grid_range[0]-1) < next_y:
            #  print "y, out_of_range!!!!"
            next_y = y
            out_of_range = True

        if next_x < 0 or (grid_range[1]-1) < next_x:
            #  print "x, out of range!!!!!"
            next_x = x
            out_of_range = True

        collision = False
        if grid[next_y, next_x] == -1:
            #  print "collision!!!!!"
            collision = True
            #  if action == 0 or action == 1:
                #  next_x = x
            #  elif action == 2 or action == 3:
                #  next_y = y

        return [next_y, next_x], out_of_range, collision


    def get_next_orientation(self, state, orientation, action, agent_id):
        if state != self.goal[agent_id]:
            next_state, out_of_range, collision = self.move(state, action)
            if not out_of_range or not collision:
                diff_y = next_state[0] - state[0]
                diff_x = next_state[1] - state[1]
                orientation = math.atan2(diff_y, diff_x)
                #  print "orientation : ", math.degrees(orientation)
                #  if orientation < 0.0:
                    #  orientation = 2.0*math.pi + orientation
                #  print "orientation : ", math.degrees(orientation)
        return orientation

    def get_next_state_and_probs(self, state, action, goal):
        transition_probability = 1 - self.noise
        probs = np.zeros([self.n_action])
        probs[action] = transition_probability
        probs += self.noise / self.n_action
        
        next_state_list = []
        next_collision_list = []

        for a in xrange(self.n_action):
            if state != goal:
                next_state, out_of_range, collision = self.move(state, a)
                next_state_list.append(next_state)
                next_collision_list.append(collision)
                self.collisions_ = next_collision_list

                if out_of_range:
                    probs[self.n_action-1] += probs[a]
                    probs[a] = 0
            else:
                next_state = state

                if a != self.n_action - 1:
                    probs[self.n_action-1] += probs[a]
                    probs[a] = 0
                next_state_list.append(next_state)

        return next_state_list, probs

    def get_transition_matrix(self):
        P = np.zeros((self.n_state, self.n_state, self.n_action), dtype=np.float32)
        for state_index in xrange(self.n_state):
            state = self.index2state(state_index)
            #  print "state : ", state
            for action_index in xrange(self.n_action):
                action = self.action_list[action_index]
                #  print "action : ", action

                next_state_list, probs = self.get_next_state_and_probs(state, action)
                #  print "next_state_list : ", next_state_list
                #  print "probs : ", probs
                for i in xrange(len(probs)):
                    next_state = next_state_list[i]
                    #  print "next_state : ", next_state
                    next_state_index = self.state2index(next_state)
                    probability = probs[i]
                    #  print "probability : ", probability
                    P[state_index, next_state_index, action_index] = probability
        #  print "P : "
        #  print P
        #  print P.shape
        return P

    def reset(self, start_position={0: [0, 0], 1: [4, 4]}, goal_position={0: [4, 4], 1: [0, 0]}, \
            start_orientation={0: 0.0, 1: 0.0}, orientation_list=None, random=False):
        #  self.set_start_and_goal_cross_scenario()
        if not random:
            self.set_start(start_position)
            self.set_goal(goal_position)
            self.set_orientation(start_orientation)
            self.get_relative_orientation()
        else:
            self.set_start_random()
            self.set_goal_random()
            self.set_orientation_random(orientation_list=orientation_list)
            self.get_relative_orientation()
        #  self.set_objects()
        self.agent_collision = False

        return self._state

    def step(self, action):
        if len(action) != self.num_agent:
            sys.stderr.write('Error occurred!')

        next_state_list = {}
        probs = {}
        reward = {}
        episode_end = {}
        for i in xrange(self.num_agent):
            next_state_list[i], probs[i] = \
                    self.get_next_state_and_probs(self._state[i], action[i], self.goal[i])
            self._orientation[i] = self.get_next_orientation(self._state[i], self._orientation[i], action[i], i)

            #  print next_state_list
            #  print probs

            random_num = np.random.rand()
            
            action_index = 0
            for j in xrange(len(probs[i])):
                random_num -= probs[i][j]
                if random_num < 0:
                    action_index = j
                    break
            #  print "next_state_list[i] : ", next_state_list[i]
            #  print "probs[i] : ", probs[i]
            self._state[i] = next_state_list[i][action_index]
        self.set_agent_grid()
        self.set_another_agent_position_with_grid()

        self.get_relative_orientation()
        
        reward = self.reward_function(self._state, self.goal)

        episode_end = self.check_episode_end(self._state, self.goal)

        return self._state, reward, episode_end, {}

    def check_episode_end(self, state, goal):
        episode_end = {}
        for i in xrange(self.num_agent):
            episode_end[i] = 0
        
        for i in xrange(self.num_agent):
            if state[i] == goal[i]:
                episode_end[i] = 1
         
        for i in xrange(len(state)):
            for j in xrange(i+1, len(state)):
                if state[i] == state[j]:
                    for k in xrange(self.num_agent):
                        episode_end[k] = 2
                        self.agent_collision = True

        #  print "episode_end : ", episode_end
        return episode_end

    def reward_function(self, state, goal):
        reward = {}
        for i in xrange(self.num_agent):
            reward[i] = 0.0
        
        for i in xrange(self.num_agent):
            if state[i] == goal[i]:
                reward[i] = 1.0
         
        for i in xrange(len(state)):
            for j in xrange(i+1, len(state)):
                if state[i] == state[j]:
                    for k in xrange(self.num_agent):
                        reward[k] = -0.25
        return reward

    def show_objectworld_with_state(self):
        grid = copy.deepcopy(self.grid)
        for i in xrange(self.num_agent):
            grid[tuple(self.goal[i])] = i+7
            if self._state[i] != None:
                grid[tuple(self._state[i])] = i+2
        for row in grid:
            print "|",
            for i in row:
                print "%2d" % i,
            print "|"



if __name__=="__main__":
    rows = 5
    cols = 5
    noise = 0.0
    n_objects = 12
    num_agent = 2
    seed = 2
    mode = 1

    ow = Objectworld(rows, cols, n_objects, num_agent, noise, seed=seed, mode=mode)

    print ow.grid

    print "ow.n_state : ", ow.n_state
    print "ow.n_action : ", ow.n_action

    print "ow._state : ", ow._state
    
    #  ow.set_start_random()
    #  ow.set_goal_random()

    #  count = 0
    #  state_list = {}
    #  for i in xrange(ow.rows):
        #  for j in xrange(ow.cols):
            #  state_list[count] = [i, j]
            
            #  print "%2d " % count, 
            #  count += 1
        #  print ""

    #  print state_list

    #  for i in xrange(ow.n_state):
        #  print ow.index2state(i)
        #  print ow.state2index(state_list[i])
    orientation_list = [-180, -135, -90, -45, 0, 45, 90, 135, 180]

    for i in xrange(10):
        print "================================="
        print "episode : ", i
        observation = ow.reset(orientation_list=orientation_list, random=True)
        print "ow.start : ", ow.start
        print "ow.goal : ", ow.goal
        ow.set_objects()
        for j in xrange(10):
            print "-----------------------------"
            print "step : ", j
            print "state : ", observation
            print "ow._orientation(before) : ", ow._orientation
            print "ow.relative_orientation(before) : ", ow.relative_orientation
            ow.show_objectworld_with_state()
            for agent_id in xrange(ow.num_agent):
                print "ow.agent_grid[", agent_id, "] : "
                print ow.agent_grid[agent_id]
            for agent_id in xrange(ow.num_agent):
                print "ow.another_agent_position_with_grid[", agent_id, "] : "
                print ow.another_agent_position_with_grid[agent_id]
            action = ow.get_sample_action()
            print "action : ", action
            ow.get_relative_velocity_vector(action)
            print "ow.relative_velocity_vector : ", ow.relative_velocity_vector
            observation, reward, episode_end, info = ow.step(action)
            print "ow._orientation : ", ow._orientation
            print "ow.relative_orientation : ", ow.relative_orientation
            print "next_state : ", observation
            print "reward : ", reward
            print "episode_end : ", episode_end

            if episode_end[0] or episode_end[1]:
                break
