#!/usr/bin.env python
#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from gridworld import Gridworld
from irl_lp import IRL_linearprograming


def normalize(vals):
  min_val = np.min(vals)
  max_val = np.max(vals)
  return (vals - min_val) / (max_val - min_val)


def heatmap_2d(input_array, title):
    plt.imshow(input_array, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    for y in range(input_array.shape[0]):
      for x in range(input_array.shape[1]):
        plt.text(x, y, '%.2f' % input_array[y, x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )

    plt.ion()
    print 'press enter to continue'
    plt.show()
    raw_input()


def heatmap_3d(input_array, title=''):
    data_2d = input_array

    data_array = np.array(data_2d)

    fig = plt.figure()
    #  ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)
    plt.title(title)

    x_data, y_data = np.meshgrid(np.arange(data_array.shape[0]), np.arange(data_array.shape[1]))
    #  print "x_data : "
    #  print x_data
    #  print "y_data : "
    #  print y_data

    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = data_array.flatten()
    ax.bar3d(x_data, y_data, np.zeros(len(z_data)), 1, 1, z_data, color='#bb33ff', alpha=1.0)
    """
    ax.bard3d(x, y, z, dx, dy, dz, color, ....)
    三次元に棒グラフを表示する関数
    引数のx, y, zは棒グラフを始める点の座標
    dx, dy, dzは棒グラフの幅を指定するパラメータ
    colorはhtlmのhex colorかmatplotlibのcolormapかmatplotlib.colorsで定義されている色で指定する。
    alphaもある
    """

    plt.show()
    raw_input()



def main(R_max, gamma, lambd, noise):
    rows = 5
    cols = 5
    #  rows = 3
    #  cols = 3
    #  rows = 2
    #  cols = 2
    #  rows = 1
    #  cols = 2

    gw = Gridworld(rows, cols, R_max)
    num_state = gw.n_state
    print "num_state : ", num_state
    num_action = gw.n_action
    print "num_action : ", num_action
    
    print "grid : "
    print gw.grid

    P_a = np.zeros((num_state, num_state, num_action))
    #  print "P_a : "
    #  print P_a

    for state_index in xrange(num_state):
        state = gw.index2state(state_index)
        #  print "====================================="
        for action in xrange(num_action):
            #  print "------------------------------------"
            #  print "state : ", state
            #  print "action : ", action
            next_state_list, probs = gw.get_next_state_and_probs(state, action, noise)
            for i in xrange(len(probs)):
                next_state = next_state_list[i]
                #  print "next_state : ", next_state
                next_state_index = gw.state2index(next_state)
                #  print "state_index : ", state_index
                #  print "next_state_index : ", next_state_index
                probability = probs[i]
                #  print "probability : ", probability
                P_a[state_index, next_state_index, action] = probability

    print "P_a : "
    print P_a
    
    Right, Left, Down, Up, Noop = gw.action_list

    policy = np.array([
        [Right, Right, Right, Down , Down],
        [Down , Right, Down , Down , Down],
        [Down , Right, Right, Down , Down],
        [Right, Right, Right, Right, Down],
        [Right, Right, Right, Right, Noop]
    ])
    #  policy = np.array([
        #  [Down , Right, Right, Right, Down],
        #  [Down , Down , Right, Down , Down],
        #  [Down , Down , Down , Down , Down],
        #  [Down , Right, Right, Down , Down],
        #  [Right, Right, Right, Right, Noop]
    #  ])
    #  policy = np.array([
        #  [Right, Right, Right, Right, Down],
        #  [Right, Right, Right, Right, Down],
        #  [Right, Right, Right, Right, Down],
        #  [Right, Right, Right, Right, Down],
        #  [Right, Right, Right, Right, Noop]
    #  ])
    #  policy = np.array([
        #  [Right, Down , Down],
        #  [Right, Right, Down],
        #  [Right, Right, Noop]
    #  ])
    #  policy = np.array([
        #  [Right, Right, Down],
        #  [Right, Right, Down],
        #  [Right, Right, Noop]
    #  ])
    #  policy = np.array([
        #  [Right, Down],
        #  [Right, Noop]
    #  ])
    #  policy = np.array([
        #  [Right, Noop]
    #  ])

    print "policy : "
    gw.show_policy(policy.reshape(-1))
    policy = np.transpose(policy).reshape(-1)
    print policy


    print "#################################################"
    lp_irl = IRL_linearprograming(P_a, policy, gamma, lambd, R_max)

    reward = lp_irl.optimize()
    reward = normalize(reward) * R_max
    reward = np.transpose(reward.reshape((gw.rows, gw.cols)))
    print "reward map :"
    print reward

    heatmap_2d(reward, 'reward_map')

    heatmap_3d(reward, 'reward_map(3D)')


if __name__=="__main__":
    R_max = 10.0
    gamma = 0.5
    lambd = 10.0

    noise = 0.3

    main(R_max, gamma, lambd, noise)
