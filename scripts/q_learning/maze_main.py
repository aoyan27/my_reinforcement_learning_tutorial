#!/usr/bin/env python
#coding:utf-8
"""
迷路探索
   5x5の大きさの迷路を探索する
   状態 : 自身のマップ上の位置(Q_tableはマップ全体に確保される)
   行動 : 上下左右の４パターン(上 : a=0 左 : a=1 下 : a=2 右 : a=3)
   報酬 : ゴールに到達した時
"""

import numpy as np
import sys
sys.path.append('../../')
from envs.maze import Maze
from agents.maze_agent import Agent

def main():
    start = (1, 1)
    rows = 10
    cols = 9
    goal = (8, 6)
    obstacle_num = 9
    env = Maze(rows, cols, start, goal, obstacle_num)
    
    action_num = 4
    agent = Agent(env.maze, action_num)

    max_episode = 3000
    max_step = 1000

    state = start
    
    success_num = 0
    failure_num = 0

    env.display()
    for i in xrange(max_episode):
        #  print "Episode : ", i
        state = env.reset()
        #  env.display()
        for j in xrange(max_step):
            #  print "step : ", j
            action = agent.epsilon_greedy(state)
            next_state, reward, episode_end = env.step(action)
            #  print "state : ", state, " action : ", action, " next_state : ", next_state, " reward : ", reward, " episode_end : ", episode_end
            #  env.display_action()

            if not episode_end:
                #  print "Q update!!!!!! "
                agent.q_update(state, next_state, action, reward)
                state = next_state
                if reward > 0:
                    success_num += 1
                    break
            else:
                failure_num += 1
                break

    agent.display_q_table()
    print "Success : ", success_num
    print "Failure : ", failure_num

    agent.show_policy()


if __name__ == "__main__":
    main()
