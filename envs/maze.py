#!/usr/bin/env python
#coding: utf-8
"""
迷路探索
   5x5の大きさの迷路を探索する
   状態 : 自身のマップ上の位置(Q_tableはマップ全体に確保される)
   行動 : 上下左右の４パターン(上 : a=0 左 : a=1 下 : a=2 右 : a=3)
   報酬 : ゴールに到達した時
"""
import sys
import numpy as np
import random
import copy

class Maze:
    def __init__(self, rows, cols, start, goal, obstacle_num):
        self.start = start
        self.goal = goal
        self.rows = rows
        self.col = cols
        self.maze = np.zeros((self.col+2, self.rows+2), dtype=np.int32)
        self.create_frame()
        self.set_start(self.start)
        self.set_goal(self.goal)
        self.set_obstacles(obstacle_num)

        self.__init_maze = copy.deepcopy(self.maze)

        self.__state = [start[0], start[1]]

    def create_frame(self):
        for i in xrange(len(self.maze)):
            for j in xrange(len(self.maze[i])):
                if i == 0 or i == (self.col+1):
                    self.maze[i][j] = -1
                else:
                    self.maze[i][0] = -1
                    self.maze[i][self.rows+1] = -1
    
    def set_start(self, start):
        self.maze[start[0]][start[1]] = 0

    def set_goal(self, goal):
        self.maze[goal[0]][goal[1]] = 10

    def set_obstacles(self, obstacle_num):
        clear_index = np.array(np.where(self.maze == 0))
        index = range(len(clear_index[0]))
        np.random.shuffle(index)

        if obstacle_num >= len(index):
            sys.stderr.write("Error!! obstacle_num is too large.\n")
            sys.exit(1)
        for i in xrange(obstacle_num):
            self.maze[clear_index[0][index[i]]][clear_index[1][index[i]]] = -1

    def display(self):
        print self.maze

    def reset(self):
        self.maze = copy.deepcopy(self.__init_maze)
        self.__state = self.start
        return self.__state

    def get_state(self, action):
        next_state = []
        x = self.__state[0]
        y = self.__state[1]

        if action == 0:
            next_state = [x-1, y]
        if action == 1:
            next_state = [x, y-1]
        if action == 2:
            next_state = [x+1, y]
        if action == 3:
            next_state = [x, y+1]

        return next_state


    def step(self, action):
        next_state = self.get_state(action)
        reward = 0
        episode_end = False

        if self.maze[next_state[0]][next_state[1]] == -1:
            episode_end = True
            reward = self.maze[next_state[0]][next_state[1]]
        else:
            episode_end = False
            reward = self.maze[next_state[0]][next_state[1]]
        self.__state = next_state


        return next_state, reward, episode_end


if __name__=="__main__":
    rows = 5
    cols = 5
    start = (1,1)
    goal = (5,5)
    obstacle_num = 5
    test_maze = Maze(rows, cols, start, goal, obstacle_num)
    
    for i in xrange(10):
        print "episode : ", i
        state = test_maze.reset()
        test_maze.display()
        for j in xrange(2):
            print "step : ", j
            action = 2
            next_state, reward, episode_end = test_maze.step(action)
            print  "state : ", state, " action : ", action, " next_state : ", next_state, " reward : ", reward, " episode_end : ", episode_end
            if not episode_end:
                state = next_state
            else:
                break
