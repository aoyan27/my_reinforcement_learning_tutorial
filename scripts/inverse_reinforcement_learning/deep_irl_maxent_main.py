#!/usr/bin/env python
#coding:utf-8

import argparse

import numpy as np
np.set_printoptions(suppress=True, threshold=np.inf)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import math

from envs.gridworld import Gridworld
from envs.objectworld import Objectworld

from agents.value_iteration import ValueIterationAgent

from deep_irl_maxent import DeepMaximumEntropyIRL


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
        plt.text(x, y, '%.2f' % input_array[y, x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )
    plt.show()

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


def create_feature_map(mode, n_state, env):
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
        feat_map = np.eye(n_state)
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
        feat_map = np.zeros([n_state, 2])
        for i in xrange(n_state):
            y, x = env.index2state(i)
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
        feat_map = np.zeros([n_state, 4])
        for i in xrange(n_state):
            y, x = env.index2state(i)
            feat_map[i, 0] = y
            feat_map[i, 1] = x
            feat_map[i, 2] = env.goal[0]
            feat_map[i, 3] = env.goal[1]
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
        feat_map = np.zeros([n_state, 1])
        for i in xrange(n_state):
            y, x = env.index2state(i)
            distance = math.sqrt((y-env.goal[0])**2 + (x-env.goal[1])**2)
            if distance == 0.0:
                feat_map[i, 0] = 1.0 / 1e-6
            else:
                feat_map[i, 0] = 1.0 / distance**2
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
        feat_map = np.zeros([n_state, 2+1])
        for i in xrange(n_state):
            y, x = env.index2state(i)
            feat_map[i, 0] = y
            feat_map[i, 1] = x

            distance = math.sqrt((y-env.goal[0])**2 + (x-env.goal[1])**2)
            if distance == 0.0:
                feat_map[i, 2] = 1.0 / 1e-6
            else:
                feat_map[i, 2] = 1.0 / distance
    elif mode == 5:
        '''
        ゴールへの距離の逆数, 最近傍の静的障害物への距離の逆数
        '''
        feat_map = np.zeros([n_state, 2])
        object_list = []
        for j in xrange(n_state):
            y_object, x_object = env.index2state(j)
            if env.grid[y_object, x_object] == -1:
                object_list.append([y_object, x_object])

        for i in xrange(n_state):
            y, x = env.index2state(i)

            distance = math.sqrt((y-env.goal[0])**2 + (x-env.goal[1])**2)
            if distance == 0.0:
                feat_map[i, 0] = 1.0 / 0.9
            else:
                feat_map[i, 0] = 1.0 / distance

            object_dist_list = []
            zero_flag = False
            for j in xrange(len(object_list)):
                tmp_dist = math.sqrt((y-object_list[j][0])**2 + (x-object_list[j][1])**2)
                object_dist_list.append(tmp_dist)
                feat_map[i, 1] = min(object_dist_list)
        
        for i in xrange(feat_map.shape[1]):
            feat_map[:, i] = normalize(feat_map[:, i])
            #  feat_map[:, i] = z_score_normalize(feat_map[:, i])
        

    return feat_map


def generate_demonstration(env, policy, reward_map, n_trajs, l_traj, \
        start_position=[0,0], rand_start=False):
    trajs = []
    for i in xrange(n_trajs):
        if rand_start:
            while 1:
                start_position = [np.random.randint(0, env.rows), np.random.randint(0, env.cols)]
                if env.grid[tuple(start_position)] == 0:
                    break

        episode_traj = {"state":[], "action":[], "next_state":[], "reward":[], "done":[]}
        env.reset(start_position)
        observation = start_position

        for j in xrange(l_traj):
            #  print j
            state = observation
            episode_traj["state"].append(state)
            #  print "state : ", state
            action = policy[env.state2index(state)]
            episode_traj["action"].append(action)
            #  print "action : ", action
            observation, reward, done, _ = env.step(action, reward_map)
            next_state = observation
            #  print "next_state : ", next_state
            #  print "reward : ", reward
            episode_traj["next_state"].append(next_state)
            episode_traj["reward"].append(reward)
            episode_traj["done"].append(done)
            
            #  if done:
                #  break
            
        #  print "episode_traj : "
        #  print episode_traj
        trajs.append(episode_traj)
    #  print trajs
    return trajs


def main(rows, cols, gamma, act_noise, n_trajs, l_traj, learning_rate, n_itrs):
    n_state = rows * cols
    n_action = 5
    r_max = 1.0

    #  n_objects, seed = 6, 3
    #  n_objects, seed = 150, 2    # 30x30 (failed)
    n_objects, seed = 100, 2    # 25x25
    #  n_objects, seed = 50, 2    # 20x20
    #  n_objects, seed = 30, 2    # 10x10, 15x15のマップ
    #  n_objects, seed = 20, 1    # 10x10
    #  n_objects, seed = 15, 1

    ################### ここからは逆強化学習のための前処理 #########################
    

    object_list = [
            (0, 3), (0, 4), (0, 5), (0, 6), (0 ,7), (0, 8), (0, 9),
            (1, 0), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9),
            (2, 0), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), 
            (3, 0), (3, 1), (3, 6), (3, 7), (3, 8), (3, 9),
            (4, 0), (4, 1), (4, 6), (4, 7), (4, 8), (4, 9),
            (5, 0), (5, 1), (5, 2), (5, 7), (5, 8), (5, 9),
            (6, 0), (6, 1), (6, 2), (6, 7), (6, 8), (6, 9),
            (7, 0), (7, 1), (7, 2), (7, 3), (7, 8), (7, 9),
            (8, 0), (8, 1), (8, 2), (8, 3),
            (9, 0), (9, 1), (9, 2), (9, 3), (9, 4)
            ]
    #  object_list = [
            #  (0, 0), (0, 1), (0 ,7), (0, 8), (0, 9),
            #  (1, 0), (1, 1), (1, 2), (1, 8), (1, 9),
            #  (2, 0), (2, 1), (2, 7), (2, 8), (2, 9), 
            #  (3, 0), (3, 1), (3, 7), (3, 8), (3, 9),
            #  (4, 0), (4, 1), (4, 7), (4, 8), (4, 9),
            #  (5, 0), (5, 1), (5, 7), (5, 8), (5, 9),
            #  (6, 0), (6, 1), (6, 2), (6, 8), (6, 9),
            #  (7, 0), (7, 1), (7, 2), (7, 3), (7, 8), (7, 9),
            #  (8, 0), (8, 1), (8, 2), (8, 3),
            #  (9, 0), (9, 1), (9, 2), (9, 3), (9, 4)
            #  ]



    #  env = Gridworld(rows, cols, r_max, act_noise)
    env = Objectworld(rows, cols, r_max, act_noise, n_objects, seed)
    #  env = Objectworld(rows, cols, r_max, act_noise, n_objects, seed, object_list=object_list, random_objects=False)
    P_a = env.get_transition_matrix()
    #  print "P_a : "
    #  print P_a
    env.show_objectworld()

    #  reward_map_gt = np.zeros([rows, cols])
    #  reward_map_gt[rows-1, cols-1] = r_max
    #  reward_map_gt[rows-1, 0] = r_max
    #  reward_map_gt[0, cols-1] = r_max
    reward_map_gt = env.grid
    
    reward_gt = reward_map_gt.transpose().reshape(n_state)
    print "reward_gt : "
    print reward_gt.reshape([rows, cols]).transpose()
    #  heatmap_2d(normalize(reward_gt.reshape([rows, cols]).transpose()), 'Reward Map(Grand truth)')
    #  heatmap_3d(normalize(reward_gt.reshape([rows, cols]).transpose()), '3D Reward Map(Grand truth)')


    vi_agent = ValueIterationAgent(env, P_a, gamma)
    vi_agent.train(reward_gt)
    print "V : "
    print vi_agent.V.reshape([rows, cols]).transpose()
    #  heatmap_2d(normalize(vi_agent.V.reshape([rows, cols]).transpose()), 'State value(Ground truth)')
    #  heatmap_3d(normalize(vi_agent.V.reshape([rows, cols]).transpose()), \
            #  '3D State value(Ground truth)')
    vi_agent.get_policy(reward_gt)
    print "policy : "
    print vi_agent.policy.reshape([rows, cols]).transpose()
    #  print vi_agent.policy.reshape(-1)
    env.show_policy(vi_agent.policy.reshape(-1))


    np.random.seed(seed)
    demo = generate_demonstration(env, vi_agent.policy, reward_gt, n_trajs, l_traj)
    #  demo = generate_demonstration(env, vi_agent.policy, reward_gt, n_trajs, l_traj, rand_start=True)
    #  print "demo : "
    #  print demo

    #  count = 0
    #  for traj in demo:
        #  print "count : "
        #  print count
        #  print "traj : "
        #  print traj
        #  count += 1
    
    ################################ ここまで ######################################

    ################### ここからが，深層逆強化学習のメイン処理 #####################
    #  feat_map = create_feature_map(0, n_state, env)
    #  feat_map = create_feature_map(1, n_state, env)
    #  feat_map = create_feature_map(2, n_state, env)
    #  feat_map = create_feature_map(3, n_state, env)
    #  feat_map = create_feature_map(4, n_state, env)
    feat_map = create_feature_map(5, n_state, env)
    print "feat_map : "
    print feat_map

    deep_maxent_irl = DeepMaximumEntropyIRL(feat_map, P_a, gamma, demo, learning_rate, n_itrs, env)
    reward = deep_maxent_irl.train()
    reward = normalize(reward)
    print "reward : "
    print reward.reshape([rows, cols]).transpose()
    heatmap_2d(reward.reshape([rows, cols]).transpose(), 'Reward Map')
    #  heatmap_3d(reward.reshape([rows, cols]).transpose(), '3D Reward Map')
    
    dirs = "/home/amsl//my_reinforcement_learning_tutorial/scripts/inverse_reinforcement_learning/models/"
    model_name = "deep_irl.model"
    deep_maxent_irl.save_model(dirs+model_name)

    ################ ここからは，ヴィジュアライズのための処理 #######################
    
    agent = ValueIterationAgent(env, P_a, gamma)
    agent.train(reward)
    print "V : "
    print agent.V.reshape([rows, cols]).transpose()
    #  print "P_a : "
    #  print P_a
    heatmap_2d(normalize(agent.V.reshape([rows, cols]).transpose()), 'State value')
    #  heatmap_3d(normalize(agent.V.reshape([rows, cols]).transpose()), '3D State value')
    #  agent.get_policy(reward)
    agent.get_policy(reward, deterministic=False)
    print "policy : "
    print agent.policy
    #  print agent.policy.reshape([rows, cols]).transpose()
    print "policy(grand truth) : "
    env.show_policy(vi_agent.policy.reshape(-1))
    print "policy : "
    #  env.show_policy(agent.policy.reshape(-1))
    env.show_policy(agent.policy, deterministic=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script is ... ')

    parser.add_argument('-r', '--rows', default=5, type=int, help='row of gridworld')
    parser.add_argument('-c', '--cols', default=5, type=int, help='column of gridworld')
    parser.add_argument('-g', '--gamma', default=0.9, type=float, help='discout factor')
    parser.add_argument('-a', '--act_noise', default=0.0, type=float, 
            help='probability of action noise')
    parser.add_argument('-t', '--n_trajs', default=200, type=int, help='number fo trajectories')
    parser.add_argument('-l', '--l_traj', default=20, type=int, help='length fo trajectory')
    parser.add_argument('-lr', '--learning_rate', default=0.01, type=float, help='learning rate')
    parser.add_argument('-ni', '--n_itrs', default=20, type=int, help='number of iterations')

    args = parser.parse_args()
    print args

    main(args.rows, args.cols, args.gamma, args.act_noise, \
            args.n_trajs, args.l_traj, args.learning_rate, args.n_itrs)

