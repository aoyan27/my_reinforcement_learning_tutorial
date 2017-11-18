#!/usr/bin/env python
#coding:utf-8

import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=np.inf)
import copy

class ValueIterationAgent:
    def __init__(self, env, P_a, gamma, limit_iteration=1000):
        self.env = env

        self.gamma = gamma

        self.limit_iteration = limit_iteration

        self.V = np.zeros(env.n_state, dtype=np.float32)

        self.P_a = P_a

        self.policy = None


    def train(self, reward_map, threshold=0.01):
        R  = reward_map
        #  print "R : "
        #  print R
        count = 0
        while 1:
            #  print "count : ", count
            V_ = copy.deepcopy(self.V)
            delta = 0
            for s in xrange(self.env.n_state):
                #  if self.env.terminal(self.env.index2state(s)):
                    #  continue
                
                state = self.env.index2state(s)
                #  print "state : ", state
                self.V[s] = max([sum([self.P_a[s, s_dash, a]*(R[s] + self.gamma*V_[s_dash]) \
                        for s_dash in xrange(self.env.n_state)]) \
                        for a in xrange(self.env.n_action)])

            #  print "self.V : "
            #  print self.V.reshape([self.env.rows, self.env.cols]).transpose()
            count += 1
            if max(abs(self.V - V_)) < threshold:
                #  print "self.P_a[0] : "
                #  print self.P_a[0]
                #  print "R : "
                #  print R
                #  print "self.V : "
                #  print self.V
                break

    def get_policy(self, reward_map, deterministic=True):
        R = reward_map
        if deterministic:
            self.policy = np.zeros([self.env.n_state])
            for s in xrange(self.env.n_state):
                state = self.env.index2state(s)
                self.policy[s] = np.argmax([sum(self.P_a[s, s_dash, a] *\
                        (R[s]+self.gamma*self.V[s_dash]) \
                        for s_dash in xrange(self.env.n_state)) \
                        for a in xrange(self.env.n_action)])
            #  print "self.policy : "
            #  print self.policy
        else:
            state_value = []
            self.policy = np.zeros([self.env.n_state, self.env.n_action])
            for s in xrange(self.env.n_state):
                state = self.env.index2state(s)
                state_value = [sum(self.P_a[s, s_dash, a] * (R[s] + self.gamma*self.V[s_dash]) \
                        for s_dash in xrange(self.env.n_state)) for a in xrange(self.env.n_action)]
                #  print "state_value : "
                #  print state_value
                if np.sum(state_value) == 0.0:
                    self.policy[s, :] = 1 / self.env.n_action
                
                else:
                    self.policy[s, :] = state_value/np.sum(state_value)

                #  print "policy : "
                #  print self.policy[s, :]
                



if __name__=="__main__":
    import sys
    sys.path.append('../')
    #  from envs.gridworld import Gridworld
    from envs.objectworld import Objectworld


    def normalize(vals):
        min_val = np.min(vals)
        max_val = np.max(vals)
        return (vals - min_val) / (max_val - min_val)



    rows = 10
    cols = 10
    R_max = 1.0

    #  noise = 0.3
    noise = 0.0

    #  n_objects, seed = 5, 1
    #  n_objects, seed = 6, 3
    #  n_objects, seed = 7, 2

    n_objects, seed = 30, 2


    object_list = [
            (0, 3), (0, 4), (0, 5), (0, 6),
            (1, 0), (1, 5), (1, 6), (1, 7),
            (2, 0), (2, 5), (2, 6), (2, 7),
            (3, 0), (3, 1), (3, 6), (3, 7), (3, 8),
            (4, 0), (4, 1), (4, 6), (4, 7), (4, 8),
            (5, 0), (5, 1), (5, 2), (5, 7), (5, 8), (5, 9),
            (6, 0), (6, 1), (6, 2), (6, 7), (6, 8), (6, 9),
            (7, 0), (7, 1), (7, 2), (7, 3), (7, 8), (7, 9),
            (8, 0), (8, 1), (8, 2), (8, 3), (8, 9),
            (9, 0), (9, 1), (9, 2), (9, 3), (9, 4)
            ]

    #  env = Gridworld(rows, cols, R_max, noise)
    #  env = Objectworld(rows, cols, R_max, noise, n_objects, seed, object_list=object_list, random_objects=False)
    env = Objectworld(rows, cols, R_max, noise, n_objects, seed)
    print env.grid
    P_a = env.get_transition_matrix()
    #  print P_a

    gamma = 0.9

    #  reward_map = np.zeros([rows, cols])
    #  reward_map[rows-1, cols-1] = R_max
    #  reward_map = np.reshape(reward_map, rows*cols)
    reward_map = normalize(env.grid)
    print "reward_map : "
    print reward_map
    reward_map = reward_map.transpose().reshape(-1)

    agent = ValueIterationAgent(env, P_a, gamma)
    print "agent.n_state : ", agent.env.n_state

    agent.train(reward_map)
    print "agent.V : "
    print agent.V.reshape([rows, cols]).transpose()

    agent.get_policy(reward_map, deterministic=False)
    #  agent.get_policy(reward_map, deterministic=True)
    print "agent.policy : "
    print agent.policy

    env.show_policy(agent.policy, deterministic=False)
    #  env.show_policy(agent.policy, deterministic=True)


    
