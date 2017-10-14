#!/usr/bin.env python
#coding:utf-8

import numpy as np
from numpy.linalg import inv
from cvxopt import matrix, solvers

class IRL_linearprograming:
    def __init__(self, P_a, policy, gamma, lambd, R_max):
        self.gamma = gamma
        self.lambd = lambd

        self.R_max = R_max

        self.P_a = P_a
        self.policy = policy

        self.n_state = len(self.P_a)
        #  print "self.n_state : ", self.n_state
        self.n_action = len(self.P_a[0][0])
        #  print "self.n_action : ", self.n_action
        
        self.num_of_constraints = \
                self.n_state*(self.n_action-1) + (self.n_action-1)*self.n_state \
                    + self.n_state + self.n_state*2 + self.n_state
        print "self.num_of_constraints : ", self.num_of_constraints
        
        self.num_of_objects = 3 * self.n_state
        print "self.num_of_objects : ", self.num_of_objects
    
    def create_constraints(self):
        A = np.zeros((self.num_of_constraints, self.num_of_objects))
        b = np.zeros(self.num_of_constraints)
        
        """
        -1 * (P_a1 - P_a)(I - gamma*P_a1).inverse() <= 0の条件の左辺の係数を表す行列を作成。
        -1をかけているのは、ソルバーが最小化しか解けないので、
        -1をかけて最小化問題に帰着させているため
        制約条件の数は、n_state*(n_action-1) --> 状態数✕ (行動数-1)となる。
        ※　行動は最適な行動と同じものは考えないので-1となる。
        """
        for s in xrange(self.n_state):
            optimal_a = self.policy[s]
            
            inverse_matrix = \
                    inv(np.identity(self.n_state) - self.gamma*self.P_a[:, :, optimal_a])
            #  print "self.P_a[:, :, optimal_a] : "
            #  print self.P_a[:, :, optimal_a]
            #  print "np.identity(self.n_state) - self.gamma*self.P_a[:, :, optimal_a] : "
            #  print np.identity(self.n_state) - self.gamma*self.P_a[:, :, optimal_a]
            #  print "inverse_matrix : "
            #  print inverse_matrix

            count = 0
            for a in xrange(self.n_action):
                if a != optimal_a:
                    A[s*(self.n_action-1)+count, :self.n_state] = \
                            -np.dot(self.P_a[s, :, optimal_a] - self.P_a[s, :, a], inverse_matrix)
                    count += 1
        #  print "A_ : "
        #  print A

        """
        min{a_j(j=1...k and a_j!=a1), (P_a1[i] - P_a[i]) * (I - gamma*P_a1).inverse() * R}という
        非線形関数minを線形問題に帰着させるために、Z_i(i=1...n)という変数を置く。
        この変数は, (P_a1[i] - P_a[i]) * (I - gamma*P_a1).inverse() * R) >= Z_i 
        (a=a1...ak and a != a1), (i=1...n)という制約条件を満たす。
        最小化問題に帰着するためにこの不等式の両辺に-1をかけるので、
        -1*(P_a1[i] - P_a[i]) * (I - gamma*P_a1).inverse() * R) + Z_i <= 0

        aは行動のパターン数、iは状態のパターン数を表す。
        そのため、制約条件の数は、n_state*(n_action-1) --> 状態数✕ (行動数-1)となる。
        ※　行動は最適な行動と同じものは考えないので-1となる。
        """
        for s in xrange(self.n_state):
            optimal_a = self.policy[s]
            
            inverse_matrix = \
                    inv(np.identity(self.n_state) - self.gamma*self.P_a[:, :, optimal_a])

            count = 0
            for a in xrange(self.n_action):
                if a != optimal_a:
                    A[self.n_state*(self.n_action-1)+s*(self.n_action-1)+count, :self.n_state] = \
                            -np.dot(self.P_a[s, :, optimal_a] - self.P_a[s, :, a], inverse_matrix)

                    A[self.n_state*(self.n_action-1)+s*(self.n_action-1)+count, self.n_state+s] = 1
                    count += 1

        #  print "A__ : "
        #  print A

        """
        目的関数は
        sum(i=1..n,min(a=1..k and a!=a1,(P_a1[i]-P_a[i])*(I-gamma*P_a1)^-1*R)-lambd*norm1(R))
        であり、絶対値の距離(norm1)が含まれている。
        絶対値を含む目的関数を線形計画問題に帰着させるには、絶対値を変数t_i(i=1...n)と置く。
        この時、変数t_iは-t_i <= x_i <= t_iを満たす。(x_iは絶対値で表されるもとの変数でnは状態数)
        ---> x_i + t_i >= 0, -x_i + t_i >= 0
        最小化問題に帰着させるため、この不等式に-1をかけて、
        -x_i - t_i <= 0, x_i - t_i <= 0
        そのため、もとの変数一つに対してプラス方向、マイナス方向にそれぞれ制約条件が発生するため、
        制約条件の数は、n_state*2
        ※　しかし、最大報酬 R_max>0 という条件から -t_i <= x_iは　0 <= x_iに置き換えられる。
        """
        for s in xrange(self.n_state):
            A[2*self.n_state*(self.n_action-1)+s, s] = 1
            A[2*self.n_state*(self.n_action-1)+s, 2*self.n_state+s] = -1

            A[2*self.n_state*(self.n_action-1)+self.n_state+s, s] = 1
            A[2*self.n_state*(self.n_action-1)+self.n_state+s, 2*self.n_state+s] = -1

        #  print "A___ : "
        #  print A

        """
        報酬関数RはR_max(>0)より小さい
        制約条件の数はn_state
        """
        for s in xrange(self.n_state):
            A[2*self.n_state*(self.n_action-1)+2*self.n_state+s, s] = 1
            b[2*self.n_state*(self.n_action-1)+2*self.n_state+s] = self.R_max
        
        #  print "A____ : "
        #  print A

        """
        最大報酬がプラスであることから伝搬する報酬は必ずプラスになるR>0
        制約条件の数はn_state
        """
        for s in xrange(self.n_state):
            A[2*self.n_state*(self.n_action-1)+2*self.n_state+self.n_state+s, s] = -1
        
        #  print "A_____ : "
        #  print A
        
        print "A : "
        print A, A.shape
        print "b : "
        print b, b.shape

        return A, b


    def create_objects(self):
        c = np.zeros(self.num_of_objects)
        """
        目的関数は、minや絶対値を含んでおり、それらをそれぞれ変数Z_i, t_iで置き換えると
        sum(i=1...n, Z_i) - lambd*(sumi=1...n, t_i)となり最小化問題に帰着させるので-1をかけて
        -1.0*sum(i=1...n, Z_i) + lambd*(sumi=1...n, t_i)
        となるので目的関数を表す行列は、
        Rに関わる0~n_state-1までは0, Zに関わるn_state~2*n_state-1までは-1, 
        tに関わる2*n_state~3*n_state-1まではlambdとなる。
        c = [0, 0, ..., 0, -1, -1, ..., -1, lambd, lambd, ..., lambd ]
        となる。
        """
        c[0:self.n_state] = 0
        c[self.n_state:2*self.n_state] = -1
        c[2*self.n_state:3*self.n_state] = self.lambd

        print "c : "
        print c, c.shape


        return c


    def optimize(self):
        A, b = self.create_constraints()
        c = self.create_objects()

        reward = solvers.lp(matrix(c), matrix(A), matrix(b))
        #  print reward['x'][:self.n_state]
        reward = np.asarray(reward['x'][:self.n_state])
        print "reward : "
        print reward
        return reward


