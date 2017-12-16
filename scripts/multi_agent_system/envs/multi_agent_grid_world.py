#!/usr/bin/env python
#coding:utf-8

import numpy as np
import sys
import copy

class Gridworld:
    def __init__(self, rows, cols, num_agent, noise, \
			start={0: [0, 0], 1: [4, 4]}, goal={0: [4, 4], 1: [0, 0]}, seed=0, mode=0):
        np.random.seed(seed)

        self.mode = mode # mode=0 : 行動４パターン, mode=1 : 行動８パターン

        self.rows = rows
        self.cols = cols

        self.num_agent = num_agent

        self.noise = noise

        self.R_max = 1.0

        self.grid = np.zeros((self.rows, self.cols))
        self.agent_grid = {0: copy.deepcopy(self.grid), 1: copy.deepcopy(self.grid)}
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
        self.set_action()

        self._state = {}

        self.start = {}
        self.start_index = None
        self.set_start(start)
        
        self.goal_index = None
        self.goal = {}
        self.set_goal(goal)
        
        self.collisions_ = []

        self.agent_collision = False
       
        self.episode_end_states = {0: 'continue', 1: 'Success', 2: 'Faild'}


    def set_action(self):
        if self.mode == 0:    # mode=0 : 行動4パターン
            self.action_list = [0, 1, 2, 3, 4]
            self.n_action = len(self.action_list)
            self.dirs = {0: '>', 1: '<', 2: 'v', 3: '^', 4: '-'}
        elif self.mode == 1:    # mode=1 : 行動8パターン
            self.action_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            self.n_action = len(self.action_list)
            self.dirs = \
					{0: '>', 1: '<', 2: 'v', 3: '^', 4: 'ur', 5: 'ul', 6: 'dr', 7: 'dl', 8: '-'}
    
    def set_agent_grid(self):
        self.agent_grid = {0: copy.deepcopy(self.grid), 1: copy.deepcopy(self.grid)}
        for i in xrange(self.num_agent):
            for j in xrange(self.num_agent):
                if i != j:
                    self.agent_grid[i][tuple(self._state[j])] = -1
    
    #  def set_start_and_goal_cross_scenario(self):
        #  center = int(self.rows / 2.0)
        #  print "center : ", center
        #  quadrant



    def set_start(self, start):
        self.start = start
        self._state = start
        self.set_agent_grid()


    def set_start_random(self, check_goal=False):
        if not check_goal:
            self.start_index = np.random.choice(xrange(self.n_state), self.num_agent, replace=False)
            #  print "self.start_index : ", self.start_index
            for i in xrange(self.num_agent):
                self.start[i] = self.index2state(self.start_index[i])
            self._state = self.start
            #  print "self.start", self.start
            self.set_agent_grid()
        else:
            while 1:
                self.start_index = \
						np.random.choice(xrange(self.n_state), self.num_agent, replace=False)
                for i in xrange(self.num_agent):
                    for j in xrange(self.num_agent):
                        if self.start_index[i] == self.goal_index[j]:
                            continue
                break
            for i in xrange(self.num_agent):
                self.start[i] = self.index2state(self.start_index[i])
            self._state = self.start
            #  print "self.start", self.start
            self.set_agent_grid()

    def set_goal(self, goal):
        self.goal = goal

    def set_goal_random(self, check_start=True):
        if check_start:
            while 1:
                self.goal_index = \
						np.random.choice(xrange(self.n_state), self.num_agent, replace=False)
                #  print "self.goal_index : ", self.goal_index
                if tuple(self.start_index) != tuple(self.goal_index):
                    break
            for i in xrange(self.num_agent):
                self.goal[i] = self.index2state(self.goal_index[i])
            #  print "self.goal", self.goal
        else:
            self.goal_index = np.random.choice(xrange(self.n_state), self.num_agent, replace=False)
            #  print "self.goal_index : ", self.goal_index
            for i in xrange(self.num_agent):
                self.goal[i] = self.index2state(self.goal_index[i])
            #  print "self.goal", self.goal


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
        if grid_range is None:
            grid_range = [self.rows, self.cols]
        if grid is None:
            grid = self.grid

        y, x = state
        next_y, next_x = state
        
        if self.mode == 0:
            if action == 0:
                #  right
                next_x = x + reflect*1
            elif action == 1:
                #  left
                next_x = x - reflect*1
            elif action == 2:
                #  down
                next_y = y + reflect*1
            elif action == 3:
                #  up
                next_y = y - reflect*1
            else:
                #  stay
                next_x = x
                next_y = y
        elif self.mode == 1:
            if action == 0:
                #  right
                next_x = x + reflect*1
            elif action == 1:
                #  left
                next_x = x - reflect*1
            elif action == 2:
                #  down
                next_y = y + reflect*1
            elif action == 3:
                #  up
                next_y = y - reflect*1
            elif action == 4:
                # upper right
                next_x = x + reflect*1
                next_y = y - reflect*1
            elif action == 5:
                # upper left
                next_x = x - reflect*1
                next_y = y - reflect*1
            elif action == 6:
                # down right
                next_x = x + reflect*1
                next_y = y + reflect*1
            elif action == 7:
                # down left
                next_x = x - reflect*1
                next_y = y + reflect*1
            else:
                #  stay
                next_x = x
                next_y = y
        
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
			random=False):
        #  self.set_start_and_goal_cross_scenario()
        if not random:
            self.set_start(start_position)
            self.set_goal(goal_position)
        else:
            self.set_start_random()
            self.set_goal_random()
        
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
    num_agent = 2
    seed = 2
    mode = 0

    gw = Gridworld(rows, cols, num_agent, noise, seed=seed, mode=mode)

    print gw.grid

    print "gw.n_state : ", gw.n_state
    print "gw.n_action : ", gw.n_action

    print "gw._state : ", gw._state
    
    gw.set_start_random()
    gw.set_goal_random()

    #  count = 0
    #  state_list = {}
    #  for i in xrange(gw.rows):
        #  for j in xrange(gw.cols):
            #  state_list[count] = [i, j]
            
            #  print "%2d " % count, 
            #  count += 1
        #  print ""

    #  print state_list

    #  for i in xrange(gw.n_state):
        #  print gw.index2state(i)
        #  print gw.state2index(state_list[i])

    for i in xrange(1):
        print "================================="
        print "episode : ", i
        observation = gw.reset()
        for j in xrange(10):
            print "-----------------------------"
            print "step : ", j
            print "state : ", observation
            gw.show_objectworld_with_state()
            action = gw.get_sample_action()
            print "action : ", action
            observation, reward, episode_end, info = gw.step(action)
            print "next_state : ", observation
            print "reward : ", reward
            print "episode_end : ", episode_end


