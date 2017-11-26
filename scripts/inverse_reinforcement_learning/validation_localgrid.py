#!/usr/bin/env python
#coding:utf-8

import argparse

import numpy as np
np.set_printoptions(suppress=True, threshold=np.inf)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import math

from networks.deep_irl_network import DeepIRLNetwork
from envs.gridworld import Gridworld
from envs.objectworld import Objectworld
from envs.localgrid_objectworld import LocalgridObjectworld

from agents.value_iteration import ValueIterationAgent

from keyborad_controller import KeyboardController


def normalize(vals):
  min_val = np.min(vals)
  max_val = np.max(vals)
  return (vals - min_val) / (max_val - min_val)

def z_score_normalize(vals):
    mean = np.mean(vals)
    std = np.std(vals)
    return (vals - mean) / std


def heatmap_2d(input_array, title):
    plt.imshow(input_array, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    for y in range(input_array.shape[0]):
      for x in range(input_array.shape[1]):
        plt.text(x, y, '%.3f' % input_array[y, x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )
    plt.draw()  # グラフ(画像)の描画
    plt.clf()  # 画面の初期化
    #  plt.show()

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


def create_feature_map(mode, env):
    '''
    特徴量を作成
    '''
    if mode == 0:
        '''
        デモの各軌道における特徴量は各状態における特徴量の合計としてとらえられるので...
        今回は，各状態における特徴ベクトルを，
            要素数が全状態数に等しく，値は{0 or 1}(自身の状態が1になってる)のベクトル，
        イメージとしては，全状態数が4(s=0, s=1, s=2, S=3)の場合は，
            f_0 = [1, 0, 0, 0]
            f_1 = [0, 1, 0, 0]
            f_2 = [0, 0, 1, 0]
            f_3 = [0, 0, 0, 1]
        みたいな感じ...
        '''
        feat_map = np.eye(env.l_n_state)
        #  print "feat_map : "
        #  print feat_map
    elif mode == 1:
        '''
        デモの各軌道における特徴量は各状態における特徴量の合計としてとらえられるので...
        今回は，各状態における特徴ベクトルを，各軌道の座標に設定、
        イメージは、全状態数が4(s=0[0,0], 1[1,0], 2[0,1], 3[1,1])の場合、
            f_0 = [0, 0]
            f_1 = [1, 0]
            f_2 = [0, 1]
            f_3 = [1, 1]
        みたいな感じ...
        '''
        feat_map = np.zeros([env.l_n_state, 2])
        for i in xrange(env.l_n_state):
            y, x = env.ilocal_index2state(i)
            feat_map[i, 0] = y
            feat_map[i, 1] = x
    elif mode == 2:
        '''
        デモの各軌道における特徴量は各状態における特徴量の合計としてとらえられるので...
        今回は，各状態における特徴ベクトルを，各軌道の座標とゴールの位置に設定、
        イメージは、全状態数が4(s=0[0,0], 1[1,0], 2[0,1], 3[1,1])、ゴールは3[1,1]の場合、
            f_0 = [0, 0, 1, 1]
            f_1 = [1, 0, 1, 1]
            f_2 = [0, 1, 1, 1]
            f_3 = [1, 1, 1, 1]
        みたいな感じ...
        '''
        feat_map = np.zeros([env.l_n_state, 4])
        for i in xrange(env.l_n_state):
            y, x = env.local_index2state(i)
            feat_map[i, 0] = y
            feat_map[i, 1] = x
            feat_map[i, 2] = env.local_goal[0]
            feat_map[i, 3] = env.local_goal[1]
    elif mode == 3:
        '''
        デモの各軌道における特徴量は各状態における特徴量の合計としてとらえられるので...
        今回は，各状態における特徴ベクトルを，各軌道の座標とゴールの距離を特徴量に...
        イメージは、全状態数が4(s=0[0,0], 1[1,0], 2[0,1], 3[1,1])、ゴールは3[1,1]の場合、
            f_0 = [1/d_0]
            f_1 = [1/d_1]
            f_2 = [1/d_2]
            f_3 = [1/d_3]
        みたいな感じ...
        '''
        feat_map = np.zeros([env.l_n_state, 1])
        for i in xrange(env.l_n_state):
            y, x = env.local_index2state(i)
            distance = math.sqrt((y-env.local_goal[0])**2 + (x-env.local_goal[1])**2)
            if distance == 0.0:
                feat_map[i, 0] = 1.0 / 0.9
            else:
                feat_map[i, 0] = 1.0 / distance
    elif mode == 4:
        '''
        デモの各軌道における特徴量は各状態における特徴量の合計としてとらえられるので...
        今回は，各状態における特徴ベクトルを，各軌道の座標と, そこからのゴールの距離を特徴量に...
        イメージは、全状態数が4(s=0[0,0], 1[1,0], 2[0,1], 3[1,1])、ゴールは3[1,1]の場合、
            f_0 = [0, 0, 1/d_0]
            f_1 = [1, 0, 1/d_1]
            f_2 = [0, 1, 1/d_2]
            f_3 = [1, 1, 1/d_3]
        みたいな感じ...
        '''
        feat_map = np.zeros([env.l_n_state, 2+1])
        for i in xrange(env.l_n_state):
            y, x = env.local_index2state(i)
            feat_map[i, 0] = y
            feat_map[i, 1] = x

            distance = math.sqrt((y-env.local_goal[0])**2 + (x-env.local_goal[1])**2)
            if distance == 0.0:
                feat_map[i, 2] = 1.0 / 0.9
            else:
                feat_map[i, 2] = 1.0 / distance
    elif mode == 5:
        '''
        ゴールへの距離の逆数, 最近傍の静的障害物への距離の逆数
        '''
        feat_map = np.zeros([env.l_n_state, 2])
        object_list = []
        for j in xrange(env.l_n_state):
            y_object, x_object = env.local_index2state(j)
            if env.local_grid[y_object, x_object] == -1:
                object_list.append([y_object, x_object])

        for i in xrange(env.l_n_state):
            y, x = env.local_index2state(i)

            distance = math.sqrt((y-env.local_goal[0])**2 + (x-env.local_goal[1])**2)
            if distance == 0.0:
                feat_map[i, 0] = 1.0 / 0.9
            else:
                feat_map[i, 0] = 1.0 / distance
            
            object_dist_list = []
            for j in xrange(len(object_list)):
                tmp_dist = math.sqrt((y-object_list[j][0])**2 + (x-object_list[j][1])**2)
                object_dist_list.append(tmp_dist)
                feat_map[i, 1] = min(object_dist_list)
 
        if len(object_list) == 0:
            feat_map[:, 1] = 0
            feat_map[:, 0] = normalize(feat_map[:, 0])
            #  feat_map[:, 0] = z_score_normalize(feat_map[:, 0])
        else:
            for i in xrange(feat_map.shape[1]):
                feat_map[:, i] = normalize(feat_map[:, i])
            #  feat_map[:, i] = z_score_normalize(feat_map[:, i])

    return feat_map


def main(rows, cols, act_noise, n_objects, seed, l_rows, l_cols, model_name):
    n_state = rows * cols
    n_action = 5
    r_max = 10.0

    env = LocalgridObjectworld(rows, cols, r_max, act_noise, n_objects, seed, l_rows, l_cols, [7, 7])

    reward_map = env.ow.grid.transpose().reshape(-1)

    observation = env.reset()
    print observation[1]

    feat_map = create_feature_map(5, env)
    print "feat_map : "
    print feat_map

    model = DeepIRLNetwork(feat_map.shape[1], 1)
    dirs = "/home/amsl/my_reinforcement_learning_tutorial/scripts/inverse_reinforcement_learning/models/"
    model.load_model(dirs+model_name, model)
    print "model : ", model

    kc = KeyboardController()

    env.show_global_objectworld()

    plt.ion()  # matplotlibの対話モードON(プログラム上での呼び出しは一回だけにしないと変な動作する)
    

    max_episode = 1
    max_step = 500
    for i in xrange(max_episode):
        print "================================"
        print "episode : ", i
        observation = env.reset()

        for j in xrange(max_step):
            print "-------------------------------"
            print "step : ", j

            print "state : ", observation[0]

            env.show_global_objectworld()
            print "local_map : "
            print observation[1]
            print "local_goal : ", env.local_goal

            feat_map = create_feature_map(5, env)
            #  print "feat_map : "
            #  print feat_map

            reward = normalize(model.get_reward(feat_map).data.reshape(-1))
            print "reward : "
            print reward.reshape([l_rows, l_cols]).transpose()
            heatmap_2d(reward.reshape([l_rows, l_cols]).transpose(), 'Reward Map')
        
            #  action = env.get_sample_action()
            action = kc.controller()

            print "action : ", action, "(", env.ow.dirs[action], ")"

            observation, reward_, done, info = env.step(action, reward_map)
            print "next_state : ", observation[0]
            #  print "local_map : "
            #  print observation[1]

            print "reward : ", reward_
            print "episode_end : ", done


            if done:
                break



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script is ... ')

    parser.add_argument('-r', '--rows', default=50, type=int, help='row of global gridworld')
    parser.add_argument('-c', '--cols', default=50, type=int, help='column of global gridworld')
    parser.add_argument('-a', '--act_noise', default=0.0, type=float, help='probability of action noise')
    parser.add_argument('-o', '--n_objects', default=200, type=int, help='number of objects')
    parser.add_argument('-s', '--seed', default=3, type=int, help='seed')

    parser.add_argument('-l_r', '--l_rows', default=5, type=int, help='row of local gridworld')
    parser.add_argument('-l_c', '--l_cols', default=5, type=int, help='column of local gridworld')

    parser.add_argument('-m', '--model_name', default='deep_irl.model', type=str, help='model name')

    args = parser.parse_args()
    print args

    main(args.rows, args.cols, args.act_noise, args.n_objects, args.seed, args.l_rows, args.l_cols, args.model_name)
