#!/usr/bin/env python
#coding:utf-8

import sys
import termios

class KeyboardController:
    def __init__(self):
        print "Keyboard controller!!"
        print "-------------------------"
        print "Moving around : "
        print "       i"
        print "   j   k   l"
        print "       ,"
        print "========================="
        print "i : up"
        print ", : down"
        print "j : left"
        print "l : right"
        print "k : stay"

    def character2action(self, ch):
        action = None
        if ch == 'i':
            action = 3
        elif ch == ',':
            action = 2
        elif ch == 'j':
            action = 1
        elif ch == 'l':
            action = 0
        elif ch == 'k':
            action = 4
        else:
            print "Error, Please press the specified key!!"
            action = 4
        return action


    def controller(self):
        #  標準入力のファイルディスクリプタを取得
        fd = sys.stdin.fileno()

        #  fdの端末属性をゲットする
        #  oldとnewには同じものが入る。
        #  newに変更を加えて、適応する
        #  oldは、後で元に戻すため
        old = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)

        #  new[3]はlflags
        #  ICANON(カノニカルモードのフラグ)を外す
        new[3] &= ~termios.ICANON
        #  ECHO(入力された文字を表示するか否かのフラグ)を外す
        new[3] &= ~termios.ECHO

        try:
            #  書き換えたnewをfdに適応する
            termios.tcsetattr(fd, termios.TCSANOW, new)
            #  キーボードから入力を受ける。
            #  lfalgsが書き換えられているので、エンターを押さなくても次に進む。echoもしない
            ch = sys.stdin.read(1)

        finally:
            #  fdの属性を元に戻す
            #  具体的にはICANONとECHOが元に戻る
            termios.tcsetattr(fd, termios.TCSANOW, old)
        
        action = self.character2action(ch)
        
        return action

if __name__ == "__main__":
    from envs.localgrid_objectworld import LocalgridObjectworld


    rows = cols = 50
    R_max = 1.0
    noise = 0.0
    n_objects = 100
    seed = 1

    l_rows = l_cols = 5

    lg_ow = LocalgridObjectworld(rows, cols, R_max, noise, n_objects, seed, l_rows, l_cols, [l_rows, l_cols])
    #  print "global_grid : "
    #  print lg_ow.ow.grid
    lg_ow.show_global_grid()

    #  print "local_grid"
    #  print lg_ow.local_grid

    reward_map = lg_ow.ow.grid.transpose().reshape(-1)
    #  print "reward_map : "
    #  print reward_map


    observation = lg_ow.reset()
    kc = KeyboardController()


    lg_ow.show_global_grid()
    
    max_episode = 1
    max_step = 500
    for i in xrange(max_episode):
        print "================================"
        print "episode : ", i
        observation = lg_ow.reset()

        for j in xrange(max_step):
            print "-------------------------------"
            print "step : ", j

            print "state : ", observation[0]
            #  print "local map : "
            #  print observation[1]
        
            #  action = lg_ow.get_sample_action()
            action = kc.controller()

            print "action : ", action, "(", lg_ow.ow.dirs[action], ")"

            observation, reward, done, info = lg_ow.step(action, reward_map)
            print "state : ", observation[0]
            #  print "local_map : "
            #  print observation[1]

            print "reward : ", reward
            print "episode_end : ", done

            lg_ow.show_global_grid()
            print "local_map : "
            print observation[1]

            if done:
                break

