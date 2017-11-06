#!/usr/bin/env python
#coding:utf-8

import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from envs.gridworld import Gridworld
from agents.value_iteration import ValueIterationAgent

from irl_maxent import MaximumEntropyIRL

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


def generate_demonstration(env, policy, reward_map, n_trajs, l_traj, start_position=[0,0]):
    trajs = []
    for i in xrange(n_trajs):
        #  start_position = [np.random.randint(0, env.rows), np.random.randint(0, env.cols)]
        #  start_position = [4, 4]

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
            
        #  print "episode_traj : "
        #  print episode_traj
        trajs.append(episode_traj)
    #  print trajs
    return trajs



def main(rows, cols, gamma, act_noise, n_trajs, l_traj, lr, n_itrs):
    n_states = rows * cols
    n_actions = 5
    r_max = 1.0

    #################### ここからの処理は逆強化学習のための前処理 ######################

    '''
    Ground truthのreward mapを作成している(今回は，右下が最大報酬)
    '''
    reward_map_gt = np.zeros([rows, cols])
    reward_map_gt[rows-1, cols-1] = r_max
    print "reward_map_gt : "
    print reward_map_gt
    #  heatmap_2d(reward_map_gt, 'Reward Map(Ground truth)')
    #  heatmap_3d(reward_map_gt, 'Reward Map(Ground truth)')

    
    '''
    Gridworldクラスを呼び出して，状態遷移行列を計算
    '''
    gw = Gridworld(rows, cols, r_max, act_noise) 
    P_a = gw.get_transition_matrix()
    
    '''
    Value Iteration(state value)で価値関数と方策を算出
    '''
    vi_agent = ValueIterationAgent(gw, P_a, gamma)
    vi_agent.train(reward_map_gt)
    print "V : "
    print vi_agent.V.reshape([5,5])
    #  heatmap_2d(vi_agent.V.reshape([5,5]), 'State value(Ground truth)')
    #  heatmap_3d(vi_agent.V.reshape([5,5]), 'State value(Ground truth)')
    vi_agent.get_policy(reward_map_gt)
    print "policy : "
    print vi_agent.policy
    #  print vi_agent.policy.reshape(-1)
    gw.show_policy(vi_agent.policy.reshape(-1))

    '''
    Value iterationより得られた方策をエキスパートとしてスタートランダムで，デモを作成
    '''
    np.random.seed(1)
    demo = generate_demonstration(gw, vi_agent.policy, reward_map_gt, n_trajs, l_traj)
    print "demo : "
    print demo
    
    #################################### ここまで ############################################
    
    #  ######################### ここからが，逆強化学習のメインの処理 #########################
    #  '''
    #  デモの各軌道における特徴量は各状態における特徴量の合計としてとらえられるので...
    #  今回は，各状態における特徴ベクトルを，
        #  要素数が全状態数に等しく，値は{0 or 1}(自身の状態が1になってる)のベクトル，
    #  イメージとしては，全状態数が4(s=0, s=1, s=2, S=3)の場合は，
        #  f_0 = [1, 0, 0, 0]
        #  f_1 = [0, 1, 0, 0]
        #  f_2 = [0, 0, 1, 0]
        #  f_3 = [0, 0, 0, 1]
    #  みたいな感じ...
    #  '''
    #  feat_map = np.eye(n_states)
    #  print "feat_map : "
    #  #  print feat_map
    #  print feat_map.shape
    
    #  maxent_irl = MaximumEntropyIRL(feat_map, P_a, gamma, demo, lr, n_itrs, gw)
    #  reward = maxent_irl.train()
    #  reward = normalize(reward)
    #  print reward


    


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='This script is ... ')

    parser.add_argument('-r', '--rows', default=5, type=int, help='row of gridworld')
    parser.add_argument('-c', '--cols', default=5, type=int, help='column of gridworld')
    parser.add_argument('-g', '--gamma', default=0.8, type=float, help='discout factor')
    parser.add_argument('-a', '--act_noise', default=0.0, type=float, 
            help='probability of action noise')
    parser.add_argument('-t', '--n_trajs', default=10, type=int, help='number fo trajectories')
    parser.add_argument('-l', '--l_traj', default=20, type=int, help='length fo trajectory')
    parser.add_argument('-lr', '--learning_rate', default=0.01, type=float, help='learning rate')
    parser.add_argument('-ni', '--n_itrs', default=20, type=int, help='number of iterations')

    args = parser.parse_args()
    print args

    main(args.rows, args.cols, args.gamma, args.act_noise, 
            args.n_trajs, args.l_traj, args.learning_rate, args.n_itrs)

