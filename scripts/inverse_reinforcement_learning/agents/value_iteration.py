#!/usr/bin/env python
#coding:utf-8

import numpy as np
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
                #  print "R[state] : ", R[state[0], state[1]]
                #  self.V[s] = R[state[0], state[1]] + \
                        #  self.gamma*max([sum(self.P_a[s, s_dash, a]*self.V[s_dash] \
                        #  for s_dash in xrange(self.env.n_state)) \
                        #  for a in xrange(self.env.n_action)])
                self.V[s] = max([sum([self.P_a[s, s_dash, a]*(R[s] + self.gamma*V_[s_dash]) \
                        for s_dash in xrange(self.env.n_state)]) \
                        for a in xrange(self.env.n_action)])

                #  print "sum[...] : "
                #  print [sum([self.P_a[s, s_dash, a]*(R[s] + self.gamma*V_[s_dash]) for s_dash in xrange(self.env.n_state)]) for a in xrange(self.env.n_action)]
                #  print "max : "
                #  print max([sum([self.P_a[s, s_dash, a]*(R[s] + self.gamma*V_[s_dash]) for s_dash in xrange(self.env.n_state)]) for a in xrange(self.env.n_action)])
                #  print "self.V : "
                #  print self.V
            #  print "self.V : "
            #  print self.V.reshape([5,5])
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
                state_value = [sum(self.P_a[s, s_dash, a] *\
                        (R[s]+self.gamma*self.V[s_dash]) \
                        for s_dash in xrange(self.env.n_state)) \
                        for a in xrange(self.env.n_action)]
                #  print "state_value : "
                #  print state_value
                self.policy[s, :] = state_value/np.sum(state_value)
                #  print "policy : "
                #  print policy[s, :]
                



if __name__=="__main__":
    import sys
    sys.path.append('../')
    from envs.gridworld import Gridworld
    

    rows = 5
    cols = 5
    R_max = 10.0

    noise = 0.3

    env = Gridworld(rows, cols, R_max, noise)
    P_a = env.get_transition_matrix()
    print P_a

    gamma = 0.5

    reward_map = np.zeros([rows, cols])
    reward_map[rows-1, cols-1] = R_max
    reward_map = np.reshape(reward_map, rows*cols)

    agent = ValueIterationAgent(env, P_a, gamma)
    print "agent.n_state : ", agent.env.n_state

    agent.train(reward_map)
    print "agent.V : "
    print agent.V.reshape([rows, cols])

    agent.get_policy(reward_map, deterministic=False)
    #  agent.get_policy(reward_map, deterministic=True)
    print "agent.policy : "
    print agent.policy

    env.show_policy(agent.policy, deterministic=False)
    #  env.show_policy(agent.policy, deterministic=True)


    
