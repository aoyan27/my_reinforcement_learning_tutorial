#!/usr/bin/env python
#coding:utf-8

import numpy as np
import copy

class ValueIterationAgent:
    def __init__(self, env, gamma, limit_iteration=1000):
        self.env = env

        self.gamma = gamma

        self.limit_iteration = limit_iteration

        self.V = np.zeros(env.n_state, dtype=np.float32)
        #  self.V[self.env.state2index(list(self.env.goal))] = \
                #  self.env.reward_function(list(self.env.goal))

        self.P_a = self.create_transition_matrix(self.env.n_state, self.env.n_action)

        self.policy = []



    def create_transition_matrix(self, n_state, n_action):
        P = np.zeros((n_state, n_state, n_action), dtype=np.float32)
        for state_index in xrange(n_state):
            state = self.env.index2state(state_index)
            #  print "state : ", state
            for action_index in xrange(n_action):
                action = self.env.action_list[action_index]
                #  print "action : ", action

                next_state_list, probs = self.env.get_next_state_and_probs(state, action)
                #  print "next_state_list : ", next_state_list
                #  print "probs : ", probs
                for i in xrange(len(probs)):
                    next_state = next_state_list[i]
                    #  print "next_state : ", next_state
                    next_state_index = self.env.state2index(next_state)
                    probability = probs[i]
                    #  print "probability : ", probability
                    P[state_index, next_state_index, action_index] = probability
        print "P : "
        print P
        return P

    def train(self, threshold=0.001):
        R  = self.env.reward_function
        count = 0
        while 1:
            print "count : ", count
            V_ = copy.deepcopy(self.V)
            delta = 0
            for s in xrange(self.env.n_state):
                #  if self.env.terminal(self.env.index2state(s)):
                    #  continue
                
                self.V[s] = R(self.env.index2state(s)) + \
                        self.gamma*max([sum(self.P_a[s, s_dash, a]*self.V[s_dash] \
                        for s_dash in xrange(self.env.n_state)) \
                        for a in xrange(self.env.n_action)])

            print "self.V : "
            print self.V
            count += 1
            if max(abs(self.V - V_)) < threshold:
                break

    def get_policy(self):
        R = self.env.reward_function
        for s in xrange(self.env.n_state):
            opt_action = np.argmax([sum(self.P_a[s, s_dash, a] *\
                    (R(self.env.index2state(s))+self.gamma*self.V[s_dash]) \
                    for s_dash in xrange(self.env.n_state)) \
                    for a in xrange(self.env.n_action)])
            self.policy.append(opt_action)

        self.policy = \
                np.transpose(np.asarray(self.policy).reshape((self.env.rows, self.env.cols)))
        print "self.policy : "
        print self.policy


if __name__=="__main__":
    from gridworld import Gridworld

    rows = 5
    cols = 5
    R_max = 10.0

    noise = 0.3

    env = Gridworld(rows, cols, R_max, noise)
    
    gamma = 0.5

    agent = ValueIterationAgent(env, gamma)
    print "agent.n_state : ", agent.env.n_state
    print "agent.V : "
    print agent.V
    #  print "agent.V[0][0] : "
    #  print agent.V[0][0]

    agent.train()

    agent.get_policy()

    env.show_policy(agent.policy.reshape(-1))


    
