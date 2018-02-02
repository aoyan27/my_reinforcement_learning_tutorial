#!/usr/bin/env python
#coding:utf-8

import numpy as np
import sys
import copy

class Objectworld:
    def __init__(self, rows, cols, n_objects, num_agent, noise,\
            start={0: [0, 0], 1: [4, 4]}, goal={0: [4, 4], 1: [0, 0]}, \
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
        self.agent_grid = {0: copy.deepcopy(self.grid), 1: copy.deepcopy(self.grid)}
        self.agent_grid_future = {0: copy.deepcopy(self.grid), 1: copy.deepcopy(self.grid)}
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
        self.movement = {}
        self.set_action()

        self._state = {}
        self._future_state = {}

        self.start = {}
        self.start_index = None
        self.set_start(start)
        
        self.goal_index = None
        self.goal = {}
        self.set_goal(goal)

        self.velocity = {}
        self.set_velocity({0: [0,0], 1: [0,0]})


        self.objects = []
        self.set_objects(n_objects_random=False)
        
        self.collisions_ = []

        self.agent_collision = False
       
        self.episode_end_states = {0: 'continue', 1: 'Success', 2: 'Faild'}


    def set_action(self):
        if self.mode == 0:    # mode=0 : 行動4パターン
            self.action_list = [0, 1, 2, 3, 4]
            self.n_action = len(self.action_list)
            self.dirs = {0: '>', 1: '<', 2: 'v', 3: '^', 4: '-'}
            self.movement = {0: [0, 1], 1: [0, -1], 2: [1, 0], 3: [-1, 0], 4: [0, 0]}
        elif self.mode == 1:    # mode=1 : 行動8パターン
            self.action_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            self.n_action = len(self.action_list)
            self.dirs \
                = {0: '>', 1: '<', 2: 'v', 3: '^', 4: 'ur', 5: 'ul', 6: 'dr', 7: 'dl', 8: '-'}
            self.movement \
                = {0: [0, 1], 1: [0, -1], 2: [1, 0], 3: [-1, 0], \
                   4: [-1, 1], 5: [-1, -1], 6: [1, 1], 7: [1, -1], 8: [0, 0]}

    def set_velocity(self, input_velocity):
        self.velocity = input_velocity


    def set_agent_grid_future(self):
        self.agent_grid_future \
                = {0: copy.deepcopy(self.agent_grid[0]), 1: copy.deepcopy(self.agent_grid[1])}
        for i in xrange(self.num_agent):
            for j in xrange(self.num_agent):
                if i != j:
                    self.agent_grid_future[i][tuple(self._future_state[j])] = -1
    
    def set_agent_grid(self):
        self.agent_grid = {0: copy.deepcopy(self.grid), 1: copy.deepcopy(self.grid)}
        for i in xrange(self.num_agent):
            for j in xrange(self.num_agent):
                if i != j:
                    self.agent_grid[i][tuple(self._state[j])] = -1

    def set_start(self, start):
        self.start = copy.deepcopy(start)
        self._state = copy.deepcopy(start)
        self._future_state = copy.deepcopy(start)
        self.set_agent_grid()
        self.set_agent_grid_future()

    def set_start_random(self, check_goal=False):
        start_ = {}
        while 1:
            self.start_index = \
                    np.random.choice(xrange(self.n_state), self.num_agent, replace=False)
            #  print "self.start_index : ", self.start_index
            #  print "self.goal_index : ", self.goal_index
            diff_start_and_goal_flag = np.asarray(self.start_index != self.goal_index)
            #  print "diff_start_and_goal_flag : ", diff_start_and_goal_flag
            not_obs_flag_list = []
            for i in xrange(self.num_agent):
                not_obs_flag_list.append(\
                        self.grid[tuple(self.index2state(self.start_index[i]))]!=-1)
            #  print "not_obs_flag_list : ", not_obs_flag_list
            if diff_start_and_goal_flag.all() \
                    and np.asarray(not_obs_flag_list).all():
                break

        for i in xrange(self.num_agent):
            start_[i] = self.index2state(self.start_index[i])

        #  print "start_ : ", start_
        self.set_start(start_)

    def set_goal(self, goal):
        self.goal = goal

    def set_goal_random(self, check_start=True):
        goal_ = {}
        while 1:
            self.goal_index = \
                    np.random.choice(xrange(self.n_state), self.num_agent, replace=False)
            #  print "self.goal_index : ", self.goal_index
            diff_start_and_goal_flag = np.asarray(self.start_index != self.goal_index)
            #  print "diff_start_and_goal_flag : ", diff_start_and_goal_flag
            not_obs_flag_list = []
            for i in xrange(self.num_agent):
                not_obs_flag_list.append(\
                        self.grid[tuple(self.index2state(self.goal_index[i]))]!=-1)
            #  print "not_obs_flag_list : ", not_obs_flag_list
            if diff_start_and_goal_flag.all() \
                    and np.asarray(not_obs_flag_list).all():
                break

        for i in xrange(self.num_agent):
            goal_[i] = self.index2state(self.goal_index[i])
        #  print "self.goal", self.goal
        self.set_goal(goal_)

    def set_objects(self, n_objects_random=True, no_object_list=None):
        self.objects = []
        self.grid = np.zeros([self.rows, self.cols])
        #  self.set_goal(self.goal)
        n_objects_ = None
        if n_objects_random:
            n_objects_ = np.random.randint(0, self.n_objects)
        else:
            n_objects_ = self.n_objects
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

                if no_object_list is not None:
                    for no_obs in no_object_list:
                        check_list.append(no_obs)
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
        self.set_agent_grid_future()
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
        action = np.random.randint(0, self.n_action)
        return action

    def get_sample_action(self):
        action = {}
        for i in xrange(self.num_agent):
            action[i] = np.random.randint(0, self.n_action)

        return action

    def move(self, state, action, grid_range=None, grid=None, reflect=1):
        if grid_range is None:
            grid_range = [self.rows, self.cols]
        if grid is None:
            grid = self.grid

        y, x = state
        next_y, next_x = state

        next_y = y + reflect*self.movement[action][0]
        next_x = x + reflect*self.movement[action][1]
        
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

        return next_state_list, probs, self.collisions_

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
    
    def get_future_state(self, state, action):
        future_state, out_of_range, collisions = self.move(state, action)
        return future_state

    def reset(self, start_position={0: [0, 0], 1: [4, 4]}, goal_position={0: [4, 4], 1: [0, 0]}, \
			random=False):
        #  self.set_start_and_goal_cross_scenario()
        if not random:
            self.set_start(start_position)
            self.set_goal(goal_position)
            self.set_velocity({0: [0,0], 1: [0,0]})
        else:
            self.set_start_random()
            self.set_goal_random()
            self.set_velocity({0: [0,0], 1: [0,0]})
        #  self.set_objects()
        self.agent_collision = False

        return self._state

    def step(self, action):
        if len(action) != self.num_agent:
            sys.stderr.write('Error occurred!')
            sys,exit(1)

        next_state_list = {}
        probs = {}
        collisions = {}
        reward = {}
        episode_end = {}
        for i in xrange(self.num_agent):
            next_state_list[i], probs[i], collisions[i] \
                    = self.get_next_state_and_probs(self._state[i], action[i], self.goal[i])

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
            self.velocity[i] = self.movement[action_index]
            self._future_state[i] = self.get_future_state(self._state[i], action_index)
        self.set_agent_grid()
        self.set_agent_grid_future()
        
        reward = self.reward_function(self._state, self.goal, collisions, action)

        episode_end = self.check_episode_end(self._state, self.goal, collisions, action)

        return self._state, reward, episode_end, {}

    def check_episode_end(self, state, goal, collisions, action):
        episode_end = {}
        for i in xrange(self.num_agent):
            episode_end[i] = 0
        
        for i in xrange(self.num_agent):
            if state[i] == goal[i]:
                episode_end[i] = 1
            elif collisions[i][action[i]]:
                episode_end[i] = 3

         
        for i in xrange(len(state)):
            for j in xrange(i+1, len(state)):
                if state[i] == state[j]:
                    for k in xrange(self.num_agent):
                        episode_end[k] = 2
                        self.agent_collision = True

        #  print "episode_end : ", episode_end
        return episode_end

    def reward_function(self, state, goal, collisions, action):
        reward = {}
        for i in xrange(self.num_agent):
            reward[i] = 0.0
        
        for i in xrange(self.num_agent):
            if state[i] == goal[i]:
                reward[i] = 1.0
            elif collisions[i][action[i]]:
                reward[i] = -1.0
         
        for i in xrange(len(state)):
            for j in xrange(i+1, len(state)):
                if state[i] == state[j]:
                    for k in xrange(self.num_agent):
                        reward[k] = -0.25
        return reward

    def show_objectworld_with_state(self):
        grid = copy.deepcopy(self.grid)
        for i in xrange(self.num_agent):
            grid[tuple(self.goal[i])] = i+6
            if self._state[i] != None:
                grid[tuple(self._state[i])] = i+2
        for row in grid:
            print "|",
            for i in row:
                print "%2d" % i,
            print "|"
    
    def show_array(self, input_array):
        array = copy.deepcopy(input_array)
        for row in array:
            print "|",
            for i in row:
                print "%2d" % i,
            print "|"




if __name__=="__main__":
    sys.path.append('../')
    from agents.a_star_agent import AstarAgent
    rows = cols = 20
    noise = 0.0
    n_objects = 50
    num_agent = 2
    seed = 0
    mode = 1

    ow = Objectworld(rows, cols, n_objects, num_agent, noise, seed=seed, mode=mode)

    print ow.grid

    print "ow.n_state : ", ow.n_state
    print "ow.n_action : ", ow.n_action

    print "ow._state : ", ow._state
    
    #  ow.set_goal_random(check_start=False)
    #  print "ow.goal : ", ow.goal
    #  ow.set_start_random()
    #  ow.set_start_random(check_goal=True)
    #  print "ow.start : ", ow.start
    #  ow.set_goal_random()
    #  print "ow.goal : ", ow.goal
    #  print "ow._state : ", ow._state
    #  for i in xrange(ow.num_agent):
        #  print "ow.agent_grid : "
        #  print ow.agent_grid[i]


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

    start = {0: [3, 3], 1: [11, 11]}
    goal = {0: [13, 13], 1:[0, 0]}

    action = {}
    success_times = 0
    collision_times = 0
    for i in xrange(100):
        print "================================="
        print "episode : ", i
        observation = ow.reset(random=True)
        #  observation = ow.reset(start_position=start, goal_position=goal, random=False)

        for j in xrange(100):
            #  print "-----------------------------"
            #  print "step : ", j
            #  print "state : ", observation
            #  ow.show_objectworld_with_state()
            #  print "ow.agent_grid : "
            #  for agent_id in xrange(ow.num_agent):
                #  ow.show_array(ow.agent_grid[agent_id])
                #  print "++++++++++++++++++++++++++++++"
            for agent_id in xrange(ow.num_agent):
                a_agent = AstarAgent(ow, agent_id=agent_id)
                a_agent.get_shortest_path(ow._state[agent_id], ow.agent_grid_future[agent_id])
                path_data = a_agent.show_path()
                #  print "view_path(", agent_id, ") : "
                #  a_agent.view_path(path_data['vis_path'])
                #  print "action_list : ", path_data['action_list']
                action[agent_id] = int(path_data['action_list'][0])
            #  action = ow.get_sample_action()
            #  print "action : ", action
            observation, reward, episode_end, info = ow.step(action)
            #  print "next_state : ", observation
            #  print "ow.velocity : ", ow.velocity
            #  print "ow._future_state : ", ow._future_state
            #  for agent_id in xrange(ow.num_agent):
                #  ow.show_array(ow.agent_grid_future[agent_id])
                #  print "++++++++++++++++++++++++++++++"

            #  print "reward : ", reward
            #  print "episode_end : ", episode_end
            episode_end_flag_list = []
            for i in xrange(ow.num_agent):
                episode_end_flag_list.append(episode_end[i]!=0)
            #  print "episode_end_flag_list : ", episode_end_flag_list
            if np.asarray(episode_end_flag_list).all():
                if episode_end[0] == 1:
                    success_times += 1
                if episode_end[0] == 2:
                    collision_times += 1
                break

        print "success_times : ", success_times
        print "collision_times : ", collision_times
