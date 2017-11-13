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

        self.noise = noise
        
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

        self.collisions_ = []

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
        state[1] = index / self.cols    # x
        return state

    def get_action_sample(self):
        return np.random.randint(self.n_action)

    def move(self, state, action):
        y, x = state
        next_y, next_x = state

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
            #  print "y, out_of_range!!!!"
            next_y = y
            out_of_range = True

        if next_x < 0 or (self.cols-1) < next_x:
            #  print "x, out of range!!!!!"
            next_x = x
            out_of_range = True

        collision = False
        if self.grid[next_y, next_x] == -1:
            #  print "collision!!!!!"
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
        next_collision_list = []

        for a in xrange(self.n_action):
            if state != list(self.goal):
                #  print "state : ", state
                next_state, out_of_range, collision = self.move(state, a)
                next_collision_list.append(collision)
                self.collisions_ = next_collision_list
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
    
    def show_policy(self, policy, deterministic=True):
        vis_policy = np.array([])
        if deterministic:
            for i in xrange(len(policy)):
                vis_policy = np.append(vis_policy, self.dirs[policy[i]])
                #  print self.dirs[policy[i]]
        else:
            for i in xrange(len(policy)):
                vis_policy = np.append(vis_policy, self.dirs[np.argmax(policy[i])])

        vis_policy = vis_policy.reshape((self.rows, self.cols)).transpose()
        for y in xrange(self.rows):
            for x in xrange(self.cols):
                if self.grid[y, x] != 0:
                    vis_policy[y, x] = '#'
        vis_policy[self.goal] = 'G'
        print vis_policy

    def terminal(self, state, index):
        episode_end = False
        #  if state == list(self.goal) or self.collisions_[index]:
        if state == list(self.goal):
            episode_end = True

        return episode_end

    def reset(self, start_position=[0,0]):
        self.state_ = start_position
        return self.state_

    def step(self, action, reward_map=None):
        next_state_list, probs = self.get_next_state_and_probs(self.state_, action)
        #  print "next_state_list : ", next_state_list
        #  print "probs : ", probs
        random_num = np.random.rand()
        #  print "random_num : ", random_num
        index = 0
        for i in xrange(len(probs)):
            random_num -= probs[i]
            #  print "random_num_ : ", random_num
            if random_num < 0:
                index = i
                break
        #  print "index : ", index
        #  print "next_state : ", next_state_list[index]

        self.state_ = next_state_list[index]
        #  print "self.satte_ : ", self.state_

        reward = None
        if reward_map is None:
            if self.state_ == list(self.goal):
                reward = self.R_max
            else:
                reward = 0
        else:
            reward = reward_map[self.state2index(self.state_)]
            #  print "reward : ", reward

        episode_end = self.terminal(self.state_, index)

        return self.state_, reward, episode_end, \
                {'probs':probs, 'random_num':random_num, 'collison': self.collisions_[index]}


if __name__ == "__main__":
    rows = 5
    cols = 5
    R_max = 1.0
    noise = 0.0
    n_objects = 5
    seed = 1

    ow = Objectworld(rows, cols, R_max, noise, n_objects, seed)

    print "ow.n_state : ", ow.n_state
    print "ow.n_action : ", ow.n_action

    reward_map = ow.grid.transpose().reshape(-1)
    print "reward_map : "
    print reward_map
    
    max_episode = 100
    max_step = 100
    for i in xrange(max_episode):
        print "==========================="
        print "episode : ", i
        observation = ow.reset()
        for j in xrange(max_step):
            print "----------------------"
            print "step : ", j
            state = observation
            print "state : ", state
            action = ow.get_action_sample()
            print "action : ", action, ow.dirs[action]

            observation, reward, done, info = ow.step(action, reward_map)
            next_state = observation
            print "observation : ", observation
            print "next_state : ", next_state
            print "reward : ", reward
            print "episode_end : ", done
            print "info : ", info

            if done:
                break

