#!/usr/bin/env python
#coding:utf-8

import numpy as np
from agents.value_iteration import ValueIterationAgent

class MaximumEntropyIRL:
    '''
    Ziebart et al. 2008 paper: Maximum Entropy Inverse Reinforcement Learning
        (https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf)
    ・このMaxEnt IRLは逆強化学習に最大エントロピー原理をあてはめることで，報酬の不定性を解消する
    (報酬の不定性とは，ある経路を最適(報酬が最大となる)とする報酬は，複数設定できてしまうこと...)
    ・このアルゴリズムのイメージは，...
    ・前提として，報酬はある特徴ベクトルとあるパラメータの線形結合で表される
    ---> 学習の目的は，このパラメータを学習すること！！！
    ・手順としては，...
    1. エキスパートのデモから特徴量を計算(デモの数で割って正規化すること！！)
    2. 特徴量とパラメータから，とりあえずの報酬を計算する
    3. その報酬を基に価値反復法などで方策を算出する
    4. 期待状態訪問頻度を計算する
    5. パラメータの勾配は(デモの特徴量 - 状態訪問頻度*特徴量)
    6. 勾配がしきい値以下になったら学習終了

    '''
    def __init__(self, feat_map, P_a, gamma, trajs, learning_rate, n_itrs, env):
        self.feat_map = feat_map
        self.P_a = P_a
        self.gamma = gamma
        self.trajs = trajs
        self.lr = learning_rate
        self.n_itrs = n_itrs
        self.env = env

    def get_expart_feature(self):
        '''
        エキスパートのデモから特徴量を計算する
        '''
        expart_feature = np.zeros(self.feat_map.shape[1])
        #  print "expart_feature : "
        #  print expart_feature
        n_trajs = 0
        for i in self.trajs:
            n_trajs += 1
            #  print "n_trajs : ", n_trajs
            #  print "i : ", i
            for j in xrange(len(i["state"])):
                #  print "j : ", j
                #  print "state : ", i["state"][j]
                #  print "action : ", i["action"][j]
                #  print "next_state : ", i["next_state"][j]
                #  print "reward : ", i["reward"][j]
                #  print "done : ", i["done"][j]
                #  print "state feature vector : "
                #  print self.feat_map[self.env.state2index(i["state"][j])]
                expart_feature += self.feat_map[self.env.state2index(i["state"][j])]
        #  print "expart_feature(sum) : "
        #  print expart_feature

        expart_feature = expart_feature / len(self.trajs)
        #  print "expart_feature(normalize) : "
        #  print expart_feature

        return expart_feature
    
    def state_visitation_frequencies(self, policy, deterministic=True):
        '''
        状態訪問回数を計算する(dynamic programingで...)
        '''

        n_state = self.env.n_state
        n_action = self.env.n_action

        T = len(self.trajs[0]["state"])
        

        mu = np.zeros([n_state, T])
        #  print mu

        for i in self.trajs:
            mu[self.env.state2index(i["state"][0]), 0] += 1
        mu[:, 0] = mu[:, 0] / len(self.trajs)

        for s in xrange(n_state):
            for t in xrange(T-1):
                if deterministic:
                    mu[s, t+1] = sum([mu[pre_s, t]*self.P_a[pre_s, s, int(policy[pre_s])] \
                            for pre_s in xrange(n_state)])    # エージェントがt-1秒の時の状態に至る確率とその状態からある行動による状態遷移確率の積で, 次の状態へ至る確率が計算できる(行動の選択は，逆強化学習によって得られた報酬マップを利用して価値反復によって学習した方策にしたがう)
                else:
                    mu[s, t+1] = sum([sum([mu[pre_s, t]*self.P_a[pre_s, s, a1]*policy[pre_s, a1] \
                            for a1 in xrange(n_action)]) for pre_s in xrange(n_state)])
        p = np.sum(mu, 1)
        return p



    def train(self):
        '''
        パラメータΘを一様乱数で初期化
        '''
        theta = np.random.uniform(size=(self.feat_map.shape[1]))
        print "theta : "
        print theta
        print theta.shape

        '''
        エキスパートの特徴量を計算
        '''
        expart_feat = self.get_expart_feature()
        print "expart_feat : "
        print expart_feat
        #  print np.sum(expart_feat)

        
        #  '''
        #  学習行程
        #  '''
        for itr in xrange(self.n_itrs):
            print 'iteration : {} / {}'.format(itr, self.n_itrs)
            
            '''
            現在のパラメータにしたがって，報酬関数を推定...
            '''
            reward = np.zeros(self.feat_map.shape[1])
            for i in xrange(len(self.feat_map)):
                reward += self.feat_map[i] * theta
            #  print reward

            '''
            推定した報酬関数を基に価値反復で方策を計算
            '''
            agent = ValueIterationAgent(self.env, self.P_a, self.gamma)
            agent.train(reward)
            #  print "V : "
            #  print agent.V.reshape([self.env.rows, self.env.cols])
            #  agent.get_policy(reward)
            agent.get_policy(reward, deterministic=False)
            #  print "policy : "
            #  print agent.policy
            #  self.env.show_policy(agent.policy.reshape(-1))
            _, policy = agent.V, agent.policy
            #  print policy.reshape([self.env.rows, self.env.cols])
            
            '''
            期待状態訪問回数(expected state visitation frequencies)を計算する
                (dynamic programingで計算する)
            '''
            #  svf = self.state_visitation_frequencies(policy)
            svf = self.state_visitation_frequencies(policy, deterministic=False)
            #  print "state visitation frequebceis : "
            #  print svf

            '''
            勾配の計算
            '''
            expected_feat = np.zeros(self.feat_map.shape[1])
            for i in xrange(len(self.feat_map)):
                expected_feat += self.feat_map[i] * svf
            #  print "expected_feat : "
            #  print expected_feat
            
            grad = expart_feat - expected_feat
            #  print "grad : "
            #  print grad

            '''
            パラメータΘの更新
            '''
            theta += self.lr * grad
            #  print "theta : "
            #  print theta

        reward_final = np.zeros(self.feat_map.shape[1])
        for i in xrange(len(self.feat_map)):
            reward_final += self.feat_map[i] * theta
        #  print reward_final
        return reward_final




