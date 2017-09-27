#!/usr/bin/env python
#coding:utf-8

import numpy as np

class ValueIterationAgent:
    def __init__(self, env, gamma, limit_iteration=1000):
        self.n_state = env.n_state
        self.n_action = env.n_state

        self.limit_iteration = limit_iteration

        self.value = {}


if __name__=="__main__":
    from gridworld import Gridworld

    rows = 5
    cols = 5
    R_max = 10.0

    noise = 0.3

    env = Gridworld(rows, cols, R_max, noise)
    
    gamma = 0.5

    agent = ValueIterationAgent(env, gamma)
    print "vi_agent.n_state : ", agent.n_state
