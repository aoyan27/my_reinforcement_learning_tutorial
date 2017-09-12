#!/usr/bin.env python
#coding:utf-8

import numpy as np
from numpy.linalg import inv
from sklearn.preprocessing import normalize
from scipy.optimize import linprog
import matplotlib.pyplot as plt

def heatmap(a):
    plt.imshow(a, cmap='gray', interpolation='nearest')
    plt.show()


gamma = 0.9  # 割引率

w, h = 2, 2  #状態(2*2のグリッドマップ)
#  w, h = 5, 5  #状態(2*2のグリッドマップ)
grid = np.zeros((w, h))
print "grid : "
print grid
n = grid.size
print "n : ", n
S = np.eye(n, n)  # 単位行列(状態遷移行列の元？)＜ーー状態遷移行列とは状態遷移確率を表した行列
print "S : "
print S

k = 5
A = np.eye(k)
print "A : "
print A
UP, DOWN, RIGHT, LEFT, NOOP = 0, 1, 2, 3, 4

P = np.zeros((k, n, n))
print "P : "
print P

mask = np.zeros((k, n))
### 上方向の行動が許されない位置にゼロを代入(グリッドマップ上でのgrid[0][0]~grid[0][4])
grid.fill(1)
grid[0,:] = 0
mask[UP] = grid.reshape(-1)

### 下方向の行動が許されない位置にゼロを代入(グリッドマップ上でのgrid[4][0]~grid[4][4])
grid.fill(1)
grid[-1,:] = 0
mask[DOWN] = grid.reshape(-1)

### 右方向の行動が許されない位置にゼロを代入(グリッドマップ上でのgrid[0][4], grid[1][4]~grid[4][4])
grid.fill(1)
grid[:,-1] = 0
mask[RIGHT] = grid.reshape(-1)

### 左方向の行動が許されない位置にゼロを代入(グリッドマップ上でのgrid[0][0], grid[1][0]~grid[4][0])
grid.fill(1)
grid[:,0] = 0
mask[LEFT] = grid.reshape(-1)
#  print "grid : "
#  print grid
print "mask : "
print mask

move = np.zeros((k, n, n))
print "S  : "
print S

# 行動選択確率を表した行列を上記のでマスクすることで、その行動が許されない状態ではその行動は選択される確率がゼロになる
move[UP] = np.roll(S, shift=-w, axis=0) * mask[UP]
#  print "np.roll(S, shitf=-w, axis=0) : "
#  print np.roll(S, shift=-w, axis=0)  # 軸 : 0に関してwだけ移動させた
move[DOWN] = np.roll(S, shift=+w, axis=0) * mask[DOWN]
#  print "np.roll(S, shitf=+w, axis=0) : "
#  print np.roll(S, shift=w, axis=0)  # 軸 : 0に関してwだけ移動させた
move[RIGHT] = np.roll(S, shift=+1, axis=0) * mask[RIGHT]
#  print "np.roll(S, shitf=+1, axis=0) : "
#  print np.roll(S, shift=+1, axis=0)  # 軸 : 0に関してwだけ移動させた
move[LEFT] = np.roll(S, shift=-1, axis=0) * mask[LEFT]
#  print "np.roll(S, shitf=-1, axis=0) : "
#  print np.roll(S, shift=-1, axis=0)  # 軸 : 0に関してwだけ移動させた
#  print "move"
#  print move

# 上で作ったmove(完全にランダム性がなしで移動できる場合の状態遷移行列)を元に状態遷移行列を作成している
directions = [UP, DOWN, RIGHT, LEFT]
print "P : "
print P
#  print [move[x] for x in directions if x !=UP]
#  print sum([move[x] for x in directions if x !=UP])
for d in directions:
    P[d] += sum([move[x] for x in directions if x != d])
    #  print "P[", d, "] : "
    #  print P[d]
    P[d] = normalize(P[d], axis=0, norm='l1') * 0.3
    #  print "P[", d, "]_ : "
    #  print P[d]
    P[d] += move[d] * 0.7
    #  print "P[", d, "]__ : "
    #  print P[d]
    P[d] = normalize(P[d], axis=0, norm='l1')
    #  print "P[", d, "]___ : "
    #  print P[d]

P[NOOP] = np.eye(n)
print "P : "
print P

#  グリッドワールドの左上をゴール(終端状態)として、報酬1を与えるようにしている
grid.fill(0)
#  print "grid : "
#  print grid
grid[0,-1] = 1
#  print "grid : "
#  print grid
R = grid.reshape(-1)
#  print "R :"
#  print R
Rmax = 1.0

#  ゴールへ至るまでのエキスパートの方策を設定した
policy = np.array([
    [RIGHT, NOOP],
    [UP,    UP  ],
    ]).reshape(-1)

#  policy = np.array([
    #  [RIGHT, RIGHT, RIGHT, RIGHT, NOOP],
    #  [UP,    RIGHT, RIGHT, UP,    UP  ],
    #  [UP   , UP   , UP   , UP   , UP  ],
    #  [UP   , UP   , RIGHT, UP   , UP  ],
    #  [UP   , RIGHT, RIGHT, RIGHT, UP  ],
#  ]).reshape(-1)

print "policy : "
print policy

I = np.eye(n)  # 単位行列
#  print "I : "
#  print I
nR = np.ndarray(n)  # 報酬関数(ベクトルR)
#  print "nR : "
#  print nR
J = np.ndarray((k, n, n))  # (I - gamma * P[policy[i]]).inverse()のための変数
#  print "J : "
#  print J
for a in xrange(k):
    J[a] = inv(I - gamma*P[a])  # I - gamma * P[a] を実行
print "J : "
print J



#  original
"""
#  tr = np.transpose
#  nb_constraints = n*k*(k-1) + n*(k-1)
#  print "nb_constraints : ", nb_constraints
#  A = np.zeros((nb_constraints, 2*n))
#  print "A : "
#  print A, A.shape
#  cursor = 0

#  for ai in xrange(k):
    #  for aj in xrange(k):
        #  if ai == aj:
            #  continue
        #  print "tr(P[", ai, "] - P[", aj, "]) : "
        #  print tr(P[ai] - P[aj]).dot(tr(J[ai]))
        #  A[cursor:cursor+n, 0:n] = tr(P[ai] - P[aj]).dot(tr(J[ai]))
        #  print "A"
        #  print A[cursor:cursor+n, 0:n]
        #  print "A_ : "
        #  print A
        #  cursor += n
#  print "A__ : "
#  print A

#  for i in xrange(n):
    #  a1 = policy[i]
    #  for a in xrange(k):
        #  if a == a1:
            #  continue
        #  A[cursor:cursor+n, 0:n] = tr(P[a1, :, i] - P[a, :, i]).dot(tr(J[a1]))
        #  A[cursor, n+i] = -1
        #  cursor += 1
#  print "A__() : "
#  print A

#  b = np.zeros(nb_constraints)

#  lamb = 10000.0
#  c = np.ndarray(2*n)
#  c[:n] = -lamb
#  print "c : "
#  print c
#  c[n:] = 1
#  print "c_ : "
#  print c

#  bounds = np.array( [(-Rmax, 0) for i in range(n)] + [(-1000000, 1000000) for i in range(n)] )
#  print "bounds : "
#  print bounds

#  res = linprog(c, A, b, bounds=bounds)
#  print res
#  heatmap(-res['x'][:n].reshape(w, h))
"""
