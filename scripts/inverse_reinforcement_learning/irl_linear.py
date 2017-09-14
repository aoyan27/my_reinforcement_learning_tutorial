#!/usr/bin.env python
#coding:utf-8

import numpy as np
from numpy.linalg import inv
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

class Maze:
    def __init__(self, rows, cols, R_max):
        self.rows = rows
        self.cols = cols
        self.n_state = self.rows * self.cols

        self.R_max = R_max

        self.maze = np.zeros((self.rows, self.cols))
        # +----------------> x
        # |
        # |
        # |
        # |
        # |
        # |
        # V
        # y
        self.goal = (self.rows-1, self.cols-1)
        self.maze[self.goal] = self.R_max

        self.action_list = [0, 1, 2, 3, 4]
        self.n_action = len(self.action_list)
        self.dirs = {0: '^', 1: '<', 2: 'v', 3: '>', 4: '-'}

    def state2index(self, state):
        #  state[0] : x
        #  state[1] : y
        return state[0] + self.cols * state[1]

    def index2state(self, index):
        state = [0, 0]
        state[0] = index % self.cols
        state[1] = index / self.cols
        return state

    def get_next_state_and_probs(self, state, action):
        if state != list(self.goal):
            x, y = state
            #  print "x : ", x
            #  print "y : ", y
            if action == 0:
                #  up
                y = y - 1
            elif action == 1:
                #  left
                x = x - 1
            elif action == 2:
                #  down
                y = y + 1
            elif action == 3:
                #  right
                x = x + 1
            else:
                #  stay
                x = x
                y = y
             
            if x < 0:
                x = 0
            elif x > (self.cols-1):
                x = self.cols - 1

            if y < 0:
                y = 0
            elif y > (self.rows-1):
                y = self.rows - 1

            next_state = [x, y]
            print "next_state : "
            print next_state
        else:
            next_state = state
            print "next_state_ : "
            print next_state

        probability = 1.0

        return next_state, probability
    
    def show_policy(self, policy):
        vis_policy = np.array([])
        for i in xrange(len(policy)):
            vis_policy = np.append(vis_policy, self.dirs[policy[i]])
            #  print self.dirs[policy[i]]
        print vis_policy.reshape((self.rows, self.cols))



class LinearIRL:
    def __init__(self, P_a, policy, gamma, lambd, R_max):
        self.gamma = gamma
        self.lambd = lambd

        self.R_max = R_max

        self.P_a = P_a
        self.policy = policy

        self.n_state = len(self.P_a)
        print "self.n_state : ", self.n_state
        self.n_action = len(self.P_a[0][0])
        print "self.n_action : ", self.n_action
        
        self.num_of_constraints = \
                self.n_state*(self.n_action-1) + (self.n_action-1)*self.n_state \
                    + self.n_state + self.n_state*2 + self.n_state
        print "self.num_of_constraints : ", self.num_of_constraints
        
        self.num_of_objects = 3 * self.n_state
        print "self.num_of_objects : ", self.num_of_objects
    
    def create_constraints(self):
        A = np.zeros((self.num_of_constraints, self.num_of_objects))
        #  print "A : ", A.shape
        b = np.zeros(self.num_of_constraints)
        #  print "b : ", b.shape
        
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
        非線形関数を線形問題に帰着させるために、Z_i(i=1...n)という変数を置く。
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
        
        print "A_____ : "
        print A
        print "b : "
        print b 

        return A, b


    def create_objects(self):
        c = np.zeros(self.num_of_objects)
        #  print "c : ", c.shape
        """
        木たき関数は、minや絶対値を含んでおり、それらをそれぞれ変数Z_i, t_iで置き換えると
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
        print c


        return c


    def optimize(self):
        A, b = self.create_constraints()
        c = self.create_objects()

        reward = solvers.lp(matrix(c), matrix(A), matrix(b))
        #  print reward['x'][:self.n_state]
        reward = np.asarray(reward['x'][:self.n_state])
        #  print "reward : "
        #  print reward
        #  return np.array(reward['x'])
        return reward


def normalize(vals):
  min_val = np.min(vals)
  max_val = np.max(vals)
  return (vals - min_val) / (max_val - min_val)

def heatmap_2d(input_array, title):
    plt.imshow(input_array, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    print "input_array.shape : ", input_array.shape

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



def main(R_max, gamma, lambd):
    rows = 5
    cols = 5
    #  rows = 3
    #  cols = 3
    #  rows = 2
    #  cols = 2
    #  rows = 1
    #  cols = 2

    env = Maze(rows, cols, R_max)
    num_state = env.n_state
    print "num_state : ", num_state
    num_action = env.n_action
    print "num_action : ", num_action
    
    print "maze : "
    print env.maze

    P_a = np.zeros((num_state, num_state, num_action))
    #  print "P_a : "
    #  print P_a

    for state_index in xrange(num_state):
        state = env.index2state(state_index)
        #  print "====================================="
        for action in xrange(num_action):
            #  print "------------------------------------"
            #  print "state : ", state
            #  print "action : ", action
            next_state, probability = env.get_next_state_and_probs(state, action)
            #  print "next_state : ", next_state
            next_state_index = env.state2index(next_state)
            #  print "state_index : ", state_index
            #  print "next_state_index : ", next_state_index
            #  print "probability : ", probability
            P_a[state_index, next_state_index, action] = probability

    print "P_a : "
    print P_a
    
    Up, Left, Down, Right, Noop = env.action_list

    #  policy = np.array([
        #  [Down , Right, Right, Right, Down],
        #  [Down , Down , Right, Down , Down],
        #  [Down , Down , Down , Down , Down],
        #  [Down , Right, Right, Down , Down],
        #  [Right, Right, Right, Right, Noop]
    #  ]).reshape(-1)
    policy = np.array([
        [Right, Right, Right, Right, Down],
        [Right, Right, Right, Right, Down],
        [Right, Right, Right, Right, Down],
        [Right, Right, Right, Right, Down],
        [Right, Right, Right, Right, Noop]
    ]).reshape(-1)
    #  policy = np.array([
        #  [Right, Right, Down],
        #  [Right, Right, Down],
        #  [Right, Right, Noop]
    #  ]).reshape(-1)
    #  policy = np.array([
        #  [Right, Down],
        #  [Right, Noop]
    #  ]).reshape(-1)
    #  policy = np.array([
        #  [Right, Noop]
    #  ]).reshape(-1)

    #  print "policy : "
    #  print policy
    print "policy : "
    env.show_policy(policy)


    print "#################################################"
    lp_irl = LinearIRL(P_a, policy, gamma, lambd, R_max)

    reward = lp_irl.optimize()
    reward = normalize(reward) * R_max
    reward = reward.reshape((env.rows, env.cols))
    print "reward map :"
    print reward

    heatmap_2d(reward, 'reward_map')
    




if __name__=="__main__":
    R_max = 1.0
    gamma = 0.5
    lambd = 10.0

    main(R_max, gamma, lambd)
