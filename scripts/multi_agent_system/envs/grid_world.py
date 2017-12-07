#!/usr/bin/env python
#coding:utf-8

import numpy as np
import sys

class Gridworld:
    def __init__(self, rows, cols, num_agent, noise):
        self.rows = rows
        self.cols = cols

        self.num_agent = num_agent

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
        
        self._state = {}
        for i in xrange(num_agent):
            self._state[i] = [0, 0]
        
        self._goal = {}
        for i in xrange(num_agent):
            self._goal[i] = [0, 0]
       
        self.episode_end_states = {0: 'continue', 1: 'Success', 2: 'Faild'}

        
    def state2index(self, state):
        #  state[1] : x
        #  state[0] : y
        return state[1] + self.cols*state[0]

    def index2state(self, index):
        state = [0, 0]
        state[1] = index % self.cols #  x
        state[0] = index / self.cols #  y
        return state
    
    def sample_action(self):
        action = []
        for i in xrange(self.num_agent):
            action.append(np.random.randint(0, 4))
        return action

    def move(self, state, action):
        y, x = state
        #  print "x : ", x, ", y : ", y, "(before)"
        #  print "action : ", action
        if action == 0:
            #  right
            x = x + 1
        elif action == 1:
            #  left
            x = x - 1
        elif action == 2:
            #  down
            y = y + 1
        elif action == 3:
            #  up
            y = y - 1
        else:
            # stay
            x = x
            y = y
        
        #  print "x : ", x, ", y : ", y, "(after)"
        
        out_of_range = False
        if x < 0:
            x = 0
            out_of_range = True
        elif x > (self.cols - 1):
            x = self.cols - 1
            out_of_range = True

        if y < 0:
            y = 0
            out_of_range = True
        elif y > (self.rows - 1):
            y = self.rows - 1
            out_of_range = True

        return [y, x], out_of_range

    def get_next_state_and_probs(self, state, action, goal):
        transition_probability = 1 - self.noise
        probs = np.zeros([self.n_action])
        probs[action] = transition_probability
        probs += self.noise / self.n_action
        
        next_state_list = []

        for a in xrange(self.n_action):
            if state != goal:
                next_state, out_of_range = self.move(state, a)
                next_state_list.append(next_state)

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


    def reset(self):
        #  for i in xrange(self.num_agent):
            #  self._state[i] = self.index2state(np.random.randint(self.n_state-1))
        self._state[0] = [0, 0]
        self._state[1] = [4, 4]
        
        #  self._goal[0] = [4, 4]
        #  self._goal[1] = [0, 0]
        self._goal[0] = [2, 0]
        self._goal[1] = [0, 4]

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
                    self.get_next_state_and_probs(self._state[i], action[i], self._goal[i])

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
        
        reward = self.reward_function(self._state, self._goal)

        episode_end = self.check_episode_end(self._state, self._goal)

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
    
    def render(self):
        pass
  



if __name__=="__main__":
    rows = 5
    cols = 5
    noise = 0.0
    num_agent = 2
    gw = Gridworld(rows, cols, num_agent, noise)

    print gw.grid

    print gw.n_state
    print gw.n_action

    print gw._state
    

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
            action = [2, 3]
            #  action = gw.sample_action()
            print "action : ", action
            observation, reward, episode_end, info = gw.step(action)
            print "next_state : ", observation
            print "reward : ", reward
            print "episode_end : ", episode_end


