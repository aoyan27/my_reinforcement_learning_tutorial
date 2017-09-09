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


gammma = 0.9  # 割引率

w, h = 3, 3  #状態(5*5のグリッドマップ)
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
    [RIGHT, RIGHT, NOOP],
    [UP,    RIGHT, UP  ],
    [UP,    UP   , UP  ],
    ]).reshape(-1)

print "policy : "
print policy

I = np.eye(n)
print "I : 
"
print I
