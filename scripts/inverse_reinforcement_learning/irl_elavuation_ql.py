#!/usr/bin/env python
#coding:utf-8

import numpy as np
from gridworld import Gridworld
from ql_agent import Agent


def main(rows, cols, R_max, noise):

    env = Gridworld(rows, cols, R_max, noise)

    print "env.n_state : ", env.n_state
    print "env.n_action : ", env.n_action

    gamma = 0.5
    alpha = 0.3

    agent = Agent(env.n_state, env.n_action, gamma, alpha)
    
    max_episode = 1000
    max_step = 200

    reward_map = np.load('./reward_map.npy')
    print "reward_map : "
    print  reward_map
    
    success = 0
    for i in xrange(max_episode):
        #  print "==============================================="
        observation = env.reset()
        for j in xrange(max_step):
            state = observation

            #  action = np.random.randint(env.n_action)
            action = agent.get_action(env.state2index(state))

            #  observation, reward, done, info = env.step(action)
            observation, reward, done, info = env.step(action, reward_map)
            next_state = observation
            
            agent.q_update(env.state2index(state), env.state2index(next_state), action, reward)

            #  print "episode : ", i+1, " step : ", j+1, " state : ", state, " action : ", action, " next_state : ", next_state, " reward : ", reward, " episode_end : ", done

            if done:
                success += 1
                break
    print "Success : ", success
    print "Failure : ", max_episode - success
    print "agent.q_table : "
    print agent.q_table
    
    policy = []
    for state_index in xrange(env.n_state):
        policy.append(agent.get_action(state_index, evaluation=True))

    print "policy : ", policy
    policy = np.transpose(np.asarray(policy).reshape((rows, cols)))
    print policy
    env.show_policy(policy.reshape(-1))


if __name__=="__main__":
    rows = 5
    cols = 5
    #  rows = 3
    #  cols = 3    
    #  rows = 2
    #  cols = 2
    R_max = 10.0

    noise = 0.0

    main(rows, cols, R_max, noise)


