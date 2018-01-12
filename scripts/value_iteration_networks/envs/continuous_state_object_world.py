#!/usr/bin/env python
#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
import math

import time

class Objectworld:

    def __init__(self, rows, cols, cell_size, goal, R_max, noise, n_objects, seed=None, \
            object_list=None, random_objects=True, start=[0,0], orientation=0.0, mode=0):
        np.random.seed(seed)
        
        self.mode = mode # mode=0 : 行動４パターン, mode=1 : 行動８パターン

        self.rows = rows
        self.cols = cols
        self.n_state = self.rows * self.cols
    
        self.cell_size = cell_size
        self.goal_radius = 0.1
        self.goal_distance = 0.0    
        
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
        self.orientation_ = None
        self.set_orientation(orientation)

        
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
        self.discreate_movement = {}
        self.set_action()

        self.continuous_action_list = None
        self.n_continuous_action = 0
        self.velocity_vector = {}
        self.set_continuous_action()

        self.dt = 0.05

        self.collision_ = False
        self.out_of_range_ = False


        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.continuous_x_list = []
        self.continuous_y_list = []


    def show_continuous_objectworld(self):
        start_time = time.time()

        self.ax.cla()
        
        y = self.state_[0]
        x = self.state_[1]
        theta = self.orientation_

        height, width = self.discreate2continuous(self.rows, self.cols)

        self.ax.set_ylim([0, height])
        self.ax.set_xlim([0, width])

        self.ax.set_xticks(np.arange(0, height, self.cell_size))
        self.ax.set_yticks(np.arange(0, width, self.cell_size))

        gca = plt.gca()
        gca.invert_yaxis()

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        
        self.ax.set_title('Continuous Object World')

        self.ax.grid(True)

        #  障害物エリアを生成
        tmp_objects = np.asarray(copy.deepcopy(self.objects)).transpose(1, 0)
        objects_continuous_y, objects_continuous_x \
                = self.discreate2continuous(tmp_objects[0], tmp_objects[1])
        #  objects_continuous_y += self.cell_size / 2.0
        #  objects_continuous_x += self.cell_size / 2.0
        #  self.ax.scatter(objects_continuous_x, objects_continuous_y, s=100, \
                #  color="pink", alpha=0.5, linewidths="2", edgecolors="red")

        object_rectangles = [patches.Rectangle(xy=[obs_x, obs_y], \
                width=self.cell_size, height=self.cell_size, \
                facecolor='pink', alpha=0.5, linewidth="2", edgecolor="red") \
                for (obs_x, obs_y) in zip(objects_continuous_x, objects_continuous_y)]
        for r in object_rectangles:
            self.ax.add_patch(r)

        #  ゴールエリアを生成
        c = patches.Circle(xy=(self.goal[1], self.goal[0]), radius=self.goal_radius, \
                facecolor='indigo', edgecolor='indigo', alpha=0.5)
        self.ax.add_patch(c)

        #  現在のエージェントの位置と方位を生成
        self.ax.scatter(x, y, color="red", linewidths="2", edgecolors="red")
        self.ax.plot([x, x+0.15*math.cos(theta)], [y, y+0.15*math.sin(theta)], \
                color='green', linewidth="3") 
        #  エージェントの軌道を生成
        self.continuous_x_list.append(x)
        self.continuous_y_list.append(y)
        self.ax.plot(self.continuous_x_list, self.continuous_y_list, color='blue')

        elapsed_time = time.time() - start_time
        print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

        plt.pause(.05)


    
    def set_action(self):
        if self.mode == 0:    # mode=0 : 行動4パターン
            self.action_list = [0, 1, 2, 3, 4]
            self.n_action = len(self.action_list)
            self.dirs = {0: '>', 1: 'v', 2: '<', 3: '^', 4: '-'}
            self.discreate_movement = {0: [0, 1], 1: [1, 0], 2: [0, -1], 3: [-1, 0], 4: [0, 0]}
        elif self.mode == 1:    # mode=1 : 行動8パターン
            self.action_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            self.n_action = len(self.action_list)
            self.dirs = \
                    {0: '>', 1: 'dr', 2: 'v', 3: 'dl', 4: '<', 5: 'ul', 6: '^', 7: 'ur', 8: '-'}
            self.discreate_movement \
                    = {0: [0, 1], 1: [1, 1], 2: [1, 0], 3: [1, -1], 4: [0, -1], \
                       5: [-1, -1], 6: [-1, 0], 7: [-1, 1], 8: [0, 0]}

    def set_continuous_action(self):
        self.continuous_action_list = [0, 1, 2, 3, 4, 5, 6, 7]
        self.n_continuous_action = len(self.continuous_action_list)
        self.velocity_vector \
                = {0: [0.1, -10.0], 1: [0.3, -5.0], 2: [0.5, -2.5], \
                   3: [0.6, 0.0], \
                   4: [0.5, 2.5], 5: [0.3, 5.0], 6: [0.1, 10.0], \
                   7: [0.0, 0.0]}


    def set_orientation(self, orientation):
        self.orientation_ = orientation

    def set_orientation_random(self, orientation_list=None):
        if orientation_list is None:
            self.orientation_ = np.random.rand() * 2.0*math.pi - math.pi
            #  print "self.orientation_ : ", self.orientation_
        else:
            #  print "orientation_list : ", orientation_list
            self.orientation_ = math.radians(np.random.choice(orientation_list, 1))
            #  print "self.orientation___ : ", self.orientation_
            #  print "self.orientation__ : ", math.radians(self.orientation_)
    
    def set_start(self, start):
        self.start = start
        self.state_ = start

    def set_start_random(self, check_goal=False):
        start = None
        if not check_goal:
            x = round(np.random.rand()*self.rows*self.cell_size, 3)
            y = round(np.random.rand()*self.cols*self.cell_size, 3)
            start = [y, x]
        else:
            while 1:
                x = round(np.random.rand()*self.rows*self.cell_size, 3)
                y = round(np.random.rand()*self.cols*self.cell_size, 3)
                start = [y, x]
                discreate_start = self.continuous2discreate(start[0], start[1])
                if start != self.goal and self.grid[discreate_start] != -1:
                    break

        self.set_start(start)
        #  print "self.start", self.start

    def set_goal(self, goal):
        self.goal = goal
        discreate_goal = self.continuous2discreate(self.goal[0], self.goal[1])
        self.grid[discreate_goal] = self.R_max

    def set_goal_random(self, check_start=True):
        goal = None
        if check_start:
            while 1:
                x = round(np.random.rand()*self.rows*self.cell_size, 3)
                y = round(np.random.rand()*self.cols*self.cell_size, 3)
                goal = [y, x]
                discreate_goal = self.continuous2discreate(goal[0], goal[1])
                if goal != self.start and self.grid[discreate_goal] != -1:
                    break
        else:
            x = round(np.random.rand()*self.rows*self.cell_size, 3)
            y = round(np.random.rand()*self.cols*self.cell_size, 3)
            goal = [y, x]

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
                discreate_start = self.continuous2discreate(self.start[0], self.start[1])
                discreate_goal = self.continuous2discreate(self.goal[0], self.goal[1])
                if (y, x) != discreate_start \
                        and (y, x) != discreate_goal\
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
        vis_grid = np.asarray(['-']*self.n_state).reshape(self.grid.shape)
        obstacle_index = np.where(self.grid == -1)
        vis_grid[obstacle_index] = '#'
        
        discreate_goal = self.continuous2discreate(self.goal[0], self.goal[1])
        vis_grid[discreate_goal] = 'G'
        if self.state_ != None:
            discreate_state = self.continuous2discreate(self.state_[0], self.state_[1])
            vis_grid[discreate_state] = '$'
        for row in vis_grid:
            print "|",
            for i in row:
                print "%c" % i,
            print "|"

    def state2index(self, state):
        return state[0] + self.cols*state[1]

    def index2state(self, index):
        state = [0, 0]
        state[0] = index % self.cols
        state[1] = index / self.cols
        return state

    def get_action_sample(self, continuous=True):
        action = None
        if not continuous:
            action = np.random.randint(self.n_action)
            print "action(discreate) : ", action, self.dirs[action]
            return action
        else:
            action = np.random.randint(self.n_continuous_action)
            print "action(continuous) : ", action, self.velocity_vector[action]
            return action

    def move(self, state, action, grid_range=None, reflect=1):
        if grid_range is None:
            grid_range = [self.rows, self.cols]
        y, x = state
        next_y, next_x = state

        next_y = y + reflect*self.discreate_movement[action][0]
        next_x = x + reflect*self.discreate_movement[action][1]
        
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

    def discreate2continuous(self, discreate_y, discreate_x):
        continuous_y = discreate_y * self.cell_size
        continuous_x = discreate_x * self.cell_size
        return continuous_y, continuous_x

    def continuous2discreate(self, continuous_y, continuous_x):
        discreate_y = int(continuous_y / self.cell_size)
        discreate_x = int(continuous_x / self.cell_size)
        return discreate_y, discreate_x


    def continuous_move(self, state, orientation, action, grid_range=None):
        if grid_range is None:
            grid_range = [self.rows, self.cols]

        y, x = state
        next_y, next_x = state
        yaw = orientation
        next_yaw = orientation

        linear = self.velocity_vector[action][0]
        angular = self.velocity_vector[action][1]
        print "yaw : ", math.degrees(yaw)
        next_yaw = yaw + angular*self.dt
        print "next_yaw : ", math.degrees(next_yaw)

        next_y = y + linear*math.sin(next_yaw)*self.dt
        next_x = x + linear*math.cos(next_yaw)*self.dt
        print "[next_y, next_x] :[ ", next_y, next_x, "]"

        out_of_range = False
        if next_y < 0*self.cell_size or (grid_range[0]-1)*self.cell_size < next_y:
            #  print "y, out_of_range!!!!"
            next_y = y
            out_of_range = True

        if next_x < 0*self.cell_size or (grid_range[1]-1)*self.cell_size < next_x:
            #  print "x, out of range!!!!!"
            next_x = x
            out_of_range = True

        collision = False
        next_discreate_y, next_discreate_x = self.continuous2discreate(next_y, next_x)
        if self.grid[next_discreate_y, next_discreate_x] == -1:
            #  print "collision!!!!!"
            collision = True
            #  if action == 0 or action == 1:
                #  next_x = x
            #  elif action == 2 or action == 3:
                #  next_y = y
        
        return [next_y, next_x], next_yaw, out_of_range, collision


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

    def reset(self, start_position=[0.0,0.0], start_orientation=0.0, \
            orientation_list=None, random=False):
        self.continuous_x_list = []
        self.continuous_y_list = []
        if not random:
            self.state_ = start_position
            self.orientation_ = start_orientation
        else:
            self.set_orientation_random(orientation_list=orientation_list)
            self.set_start_random()
            self.set_goal_random()
            self.state_ = self.start

        return self.state_, self.orientation_

    def terminal(self, reward):
        episode_end = False
        if reward != 0.0 or self.out_of_range_:
            episode_end = True

        return episode_end

    def get_reward(self):
        reward = 0.0
        tmp = np.asarray(self.goal) - np.asarray(self.state_ )
        self.goal_distance = np.sum(tmp**2)
        print "self.goal_distance : ", self.goal_distance
        if self.goal_distance <= self.goal_radius:
            reward = self.R_max

        if self.collision_:
            reward = -1.0

        return reward

    
    def step(self, continuous_action):
        next_state, next_orientation, out_of_range, collision \
                = self.continuous_move(self.state_, self.orientation_, continuous_action)
        print "next_state : ", next_state
        print "next_orientation : ", next_orientation
        print "out_of_range : ", out_of_range
        print "collision : ", collision

        self.state_ = next_state
        print "self.satte_ : ", self.state_
        self.orientation_ = next_orientation
        print "self.orientation_ : ", next_orientation
        self.collision_ = collision
        self.out_of_range_ = out_of_range

        reward = self.get_reward()
        episode_end = self.terminal(reward)

        return self.state_, self.orientation_, reward, episode_end, \
                {'goal_distance': self.goal_distance, \
                 'collison': self.collision_, 'out_of_range': self.out_of_range_}


if __name__ == "__main__":
    height = 10    #  単位[m]
    width = 10    #  単位[m]
    cell_size = 0.25

    rows = int(height / cell_size)
    cols = int(width / cell_size)
    print "rows, cols : ", rows, cols

    goal = [2.5, 2.5]

    R_max = 1.0
    noise = 0.0
    n_objects = 50
    seed = 1
    

    env = Objectworld(rows, cols, cell_size, goal, R_max, noise, n_objects, seed, mode=1)
    
    print "env.state_ : ", env.state_
    print "env.start : ", env.start
    print "env.goal : ", env.goal
    print "env.grid : "
    env.show_objectworld_with_state()
    
    #  for i in xrange(10):
        #  env.set_start_random()
        #  #  env.set_start_random(check_goal=True)
        #  print "env.start : ", env.start
        #  env.set_goal_random()
        #  #  env.set_goal_random(check_start=False)
        #  print "env.goal : ", env.goal
        #  env.set_objects()
        #  print "env.grid : "
        #  #  env.show_objectworld()
        #  env.show_objectworld_with_state()

    print "env.n_state : ", env.n_state
    print "env.n_action : ", env.n_action
    print "env.n_continuous_action : ", env.n_continuous_action

    max_episode = 1
    max_step = 100

    for i in xrange(max_episode):
        print "==========================="
        print "episode : ", i
        position, yaw = env.reset(random=True)
        for j in xrange(max_step):
            print "----------------------"
            print "step : ", j
            state = position
            orientation = yaw
            print "state : ", state
            print "orientation : ", orientation
            env.show_objectworld_with_state()
            env.show_continuous_objectworld()
            #  action = env.get_action_sample()
            action = env.get_action_sample(continuous=True)

            position, yaw, reward, done, info = env.step(action)
            next_state = position
            next_orientaiton = yaw

            print "next_state : ", next_state
            print "bext_orientation : ", next_orientaiton
            print "reward : ", reward
            print "episode_end : ", done
            print "info : ", info

            if done:
                break
