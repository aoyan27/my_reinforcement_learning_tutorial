#!/usr/bin/env python
#coding: utf-8
"""
チーズ製造機
    このチーズ製造機は電源状態を表す電球と2つのボタンがついています。
    ボタン1(電源)を押す( 行動1 a = 0)と電源が入り電球が点灯( 状態1 s = 0）、
    その状態からボタン2を押す( 行動2 a = 1)とチーズが生成します。
    電源が入った状態1からまたボタン1を押すと電球が消灯します( 状態2 s = 1)。 
    初期状態は電源が付いている状態(状態1)としましょう。
"""

import numpy as np

class Agent:
    ALPHA = 0.5
    GAMMA = 0.9
    EPISILON = 0.5

    def __init__(self):
        self._q_table = np.zeros((2,2), dtype=np.float32)

    def epsilon_greedy(self, state, evoluation=False):
        if not evoluation:
            if np.random.rand() < self.EPISILON:
                print "Random action"
                return int(np.round(np.random.rand()))
            else:
                print "Greedy action"
                return np.argmax(self._q_table[state])
        else:
            return np.argmax(self._q_table[state])

    def q_update(self, state, next_state, action, reward):
        self._q_table[state][action] = (1-self.ALPHA)*self._q_table[state][action] + self.ALPHA*(reward + self.GAMMA*np.max(self._q_table[next_state]))

    def display_q_table(self):
        print self._q_table

if __name__=="__main__":
    test_agent = Agent()

    state = 1
    action = test_agent.epsilon_greedy(state)
    print action
    for i in range(10):
        test_agent.q_update(0, 0, 1, 10)
        test_agent.display_q_table()
