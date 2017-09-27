#!/usr/bin/env python
#coding:utf-8

import numpy as np
from gridworld import Gridworld

def main(rows, cols, R_max, noise):
    env = Gridworld(rows, cols, R_max, noise)

    print "env.n_state : ", env.n_state
    print "env.n_action : ", env.n_action

    print env.grid
    



if __name__=="__main__":
    rows = 5
    cols = 5

    R_max = 10.0
    noise = 0.3

    main(rows, cols, R_max, noise)
