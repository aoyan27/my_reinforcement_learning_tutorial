#!/usr/bin/env python
#coding:utf-8
"""
チーズ製造機
    このチーズ製造機は電源状態を表す電球と2つのボタンがついています。
    ボタン1(電源)を押す( 行動1 a = 0)と電源が入り電球が点灯( 状態1 s = 0）、
    その状態からボタン2を押す( 行動2 a = 1)とチーズが生成します。
    電源が入った状態1からまたボタン1を押すと電球が消灯します( 状態2 s = 1)。 
    初期状態は電源が付いている状態(状態1)としましょう。
"""

import numpy as np
import sys
sys.path.append('../../')
from envs.vending_machine import VendingMachine
from vending_machine_agent import Agent

def main():
    init_state = 1
    action = 0
    reward = 0

    agent = Agent()
    env = VendingMachine(init_state)

    state = init_state
    next_state = init_state

    max_iteration = 100
    
    for i in xrange(max_iteration):
        print "i : ", i
        action = agent.epsilon_greedy(state)
        next_state, reward = env.step(action)
        agent.q_update(state, next_state, action, reward)
        
        print "state : ", state,  " action : ", action, " next_state : ", next_state, " reward : ", reward

        agent.display_q_table()
        
        state = next_state

    print "Evoluation!!!"
    state = 0
    print "State : ", state, " action : ", agent.epsilon_greedy(state, evoluation=True)
    state = 1
    print "State : ", state, " action : ", agent.epsilon_greedy(state, evoluation=True)

if __name__=="__main__":
    main()
