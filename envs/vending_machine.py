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

class VendingMachine:
    def __init__(self, state):
        self.__state = state

    def step(self, action):
        reward = 0

        if action == 0:  #電源のON、OFFを切り替えるボタンを押す行動
            if self.__state == 0:
                print "slef.__state : ", self.__state
                self.__state = 1
            else:
                self.__state = 0
        else:   #チーズを生成するボタンを押す行動
            if self.__state == 0:
                reward = 10
            else:
                reward = 0

        return self.__state, reward

        
if __name__=="__main__":
    init_state = 0
    test_vm = VendingMachine(init_state)
    
    action = 1
    state, reward = test_vm.step(action)
    print "state : ", state, " reward : ", reward


    action = 0
    state, reward = test_vm.step(action)
    print "state : ", state, " reward : ", reward
