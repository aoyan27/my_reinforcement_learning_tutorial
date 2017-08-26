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
from agents.actor_critic_maze_agent import Agent

def main():
    start = (1, 1)
    rows = 10
    cols = 9
    goal = (8, 6)
    obstacle_num = 9
    env = Maze(rows, cols, start, goal, obstacle_num)

    num_action = 4

    agent = Agent(env.maze, num_action)

    max_episode = 5000
    max_step = 100

    state = start

    env.display()

    success = 0
    failure = 0

    for i in xrange(max_episode):
        print "Episode : ", i
        #  env.display()
        state = env.reset()
        for j in xrange(max_step):
            #  print "step : ", j
            action = agent.actor(state)

            next_state, reward, episode_end = env.step(action)
            #  print "state : ", state, " action : ", action, " next_state : ", next_state, " reward : ", reward, " episode_end : ", episode_end
            
            agent.train(state, action, next_state, reward, episode_end)
            #  print "agent._mu_table : ", agent._mu_table
            #  print "agent._sigma_table : ", agent._sigma_table
            #  print "agent._v_table : ", agent._v_table

            state = next_state
            
            if episode_end:
                if reward > 0:
                    success += 1
                else:
                    failure += 1
                break
            else:
                if j >= max_step-1:
                    print "Time up!!!"
                    failure += 1

        print "Success : ", success
        print "Failure : ", failure
        print 
                
    print "agent._mu_table : ", agent._mu_table
    print "agent._sigma_table : ", agent._sigma_table
    print "agent._v_table : ", agent._v_table


    agent.show_policy()


if __name__ == "__main__":
    main()
