#!/usr/bin/env python
#coding:utf-8

import numpy as np

import copy

class Objectworld:

    def __init__(self, rows, cols, goal, R_max, noise, n_objects, seed=None, \
            object_list=None, random_objects=True, start=[0,0], mode=0):
        np.random.seed(seed)
        
        self.mode = mode # mode=0 : 行動４パターン, mode=1 : 行動８パターン

        self.rows = rows
        self.cols = cols
        self.n_state = self.rows * self.cols
        
        self.R_max = R_max

        self.noise = noise

        self.object_list = object_list
        self.random_objects = random_objects
        
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

        self.state_ = None
        
        self.start = None
        self.start_index = None
        self.set_start(start)

        self.goal = None
        self.goal_index = None
        self.set_goal(goal)

        self.n_objects = n_objects
        self.objects = []
        self.set_objects(n_objects_random=False)
        

        self.action_list = None
        self.n_action = 0
        self.dirs = {}
        self.set_action()


        self.collisions_ = []
    
    def set_action(self):
        if self.mode == 0:    # mode=0 : 行動4パターン
            self.action_list = [0, 1, 2, 3, 4]
            self.n_action = len(self.action_list)
            self.dirs = {0: '>', 1: '<', 2: 'v', 3: '^', 4: '-'}
        elif self.mode == 1:    # mode=1 : 行動8パターン
            self.action_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            self.n_action = len(self.action_list)
            self.dirs = {0: '>', 1: '<', 2: 'v', 3: '^', 4: 'ur', 5: 'ul', 6: 'dr', 7: 'dl', 8: '-'}
    
    def set_start(self, start):
        self.start = start
        self.state_ = start

    def set_start_random(self, check_goal=False):
        start = None
        if not check_goal:
            self.start_index = \
                    np.random.choice(xrange(self.n_state), 1, replace=False)
            #  print "self.start_index : ", self.start_index
        else:
            while 1:
                self.start_index = \
                        np.random.choice(xrange(self.n_state), 1, replace=False)
                if tuple(self.start_index) != tuple(self.goal_index) \
                        and self.grid[tuple(self.index2state(self.start_index))] != -1:
                    break

        start = self.index2state(self.start_index)
        self.set_start(start)
        #  print "self.start", self.start

    def set_goal(self, goal):
        self.goal = goal
        self.grid[tuple(self.goal)] = self.R_max

    def set_goal_random(self, check_start=True):
        goal = None
        if check_start:
            while 1:
                self.goal_index = \
                        np.random.choice(xrange(self.n_state), 1, replace=False)
                #  print "self.goal_index : ", self.goal_index
                if tuple(self.start_index) != tuple(self.goal_index) \
                        and self.grid[tuple(self.index2state(self.goal_index))] != -1:
                    break
        else:
            self.goal_index = \
                    np.random.choice(xrange(self.n_state), 1, replace=False)
            #  print "self.goal_index : ", self.goal_index

        goal = self.index2state(self.goal_index)
        self.set_goal(goal)
        #  print "self.goal", self.goal


    def set_objects(self, n_objects_random=True):
        self.objects = []
        self.grid = np.zeros([self.rows, self.cols])
        self.set_goal(self.goal)
        n_objects_ = None
        if n_objects_random:
            n_objects_ = np.random.randint(0, self.n_objects)
        else:
            n_objects_ = self.n_objects
        #  print "n_objects_ : ", n_objects_
        if self.random_objects:
            i = 0
            while i < n_objects_:
                #  print " i : ", i
                y = np.random.randint(0, self.rows)
                x = np.random.randint(0, self.cols)
                if (y, x) != tuple(self.start) \
                        and (y, x) != tuple(self.goal)\
                        and self.grid[y, x] != -1:
                    self.objects.append((y, x))
                    self.grid[y, x] = -1
                    i += 1
                #  print "(y, x) : ", (y, x)
        else:
            self.objects = self.object_list
        #  print self.objects

    def show_objectworld(self):
        grid_world = copy.deepcopy(self.grid)
        for row in grid_world:
            print "|",
            for i in row:
                print "%2d" % i,
            print "|"

    def show_objectworld_with_state(self):
        grid = copy.deepcopy(self.grid)
        grid[tuple(self.goal)] = 9
        if self.state_ != None:
            grid[tuple(self.state_)] = 1
        for row in grid:
            print "|",
            for i in row:
                print "%2d" % i,
            print "|"

    def state2index(self, state):
        return state[0] + self.cols*state[1]

    def index2state(self, index):
        state = [0, 0]
        state[0] = index % self.cols
        state[1] = index / self.cols
        return state

    def get_action_sample(self):
        return np.random.randint(self.n_action)

    def move(self, state, action, grid_range=None, reflect=1):
        if grid_range is None:
            grid_range = [self.rows, self.cols]
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
        if self.grid[next_y, next_x] == -1:
            #  print "collision!!!!!"
            collision = True
            #  if action == 0 or action == 1:
                #  next_x = x
            #  elif action == 2 or action == 3:
                #  next_y = y

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
            if state != self.goal:
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
                #  if collision:
                    #  probs[self.n_action-1] += probs[a]
                    #  probs[a] = 0
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
        vis_policy[tuple(self.goal)] = 'G'
        for y in xrange(self.rows):
            print "|",
            for x in xrange(self.cols):
                if self.grid[y, x] == -1:
                    vis_policy[y, x] = '#'
                    print "#",
                else:
                    print vis_policy[y, x],
            print "|"

    def terminal(self, state, index):
        episode_end = False
        if state == self.goal or self.collisions_[index]:
        #  if state == self.goal:
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
            if self.state_ == self.goal:
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
    goal = [rows-1, cols-1]

    R_max = 1.0
    noise = 0.0
    n_objects = 5
    seed = 1
    
    env = Objectworld(rows, cols, goal, R_max, noise, n_objects, seed, mode=1)
    
    print "env.state_ : ", env.state_
    print "env.start : ", env.start
    print "env.goal : ", env.goal
    print "env.grid : "
    env.show_objectworld_with_state()
    
    for i in xrange(10):
        env.set_start_random()
        env.set_goal_random()
        env.set_objects()
        print "env.grid : "
        #  env.show_objectworld()
        env.show_objectworld_with_state()

    #  print "env.n_state : ", env.n_state
    #  print "env.n_action : ", env.n_action

    #  reward_map = env.grid.transpose().reshape(-1)
    #  print "reward_map : "
    #  print reward_map
    
    #  max_episode = 1
    #  max_step = 100

    #  for i in xrange(max_episode):
        #  print "==========================="
        #  print "episode : ", i
        #  observation = env.reset()
        #  for j in xrange(max_step):
            #  print "----------------------"
            #  print "step : ", j
            #  state = observation
            #  print "state : ", state
            #  env.show_objectworld_with_state()
            #  action = env.get_action_sample()
            #  print "action : ", action, env.dirs[action]

            #  observation, reward, done, info = env.step(action, reward_map)
            #  next_state = observation
            #  print "observation : ", observation
            #  print "next_state : ", next_state
            #  print "reward : ", reward
            #  print "episode_end : ", done
            #  print "info : ", info

            #  if done:
                #  break
