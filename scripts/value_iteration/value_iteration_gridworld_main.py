#!/usr/bin/env python
#coding:utf-8

import numpy as np
from envs.gridworld import Gridworld
from agents.value_iteration import ValueIterationAgent

def main(rows, cols, R_max, noise, gamma):
    env = Gridworld(rows, cols, R_max, noise)

    print "env.n_state : ", env.n_state
    print "env.n_action : ", env.n_action

    print env.grid

    agent = ValueIterationAgent(env, gamma)

    agent.train()
    agent.get_policy()

    env.show_policy(agent.policy.reshape(-1))


if __name__=="__main__":
    rows = 5
    cols = 5

    R_max = 10.0
    noise = 0.3

    gamma = 0.5

    main(rows, cols, R_max, noise, gamma)
