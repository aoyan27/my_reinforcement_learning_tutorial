#!/usr/bin/env python
#coding:utf-8

import argparse
import sys
import copy
from collections import deque

import gym
import numpy as np

import chainer.links as L
import chainer.functions as F
from chainer import Chain, optimizers, Variable, serializers

class Network(Chain):
    def __init__(self, n_in, n_out):
        super(Network, self).__init__(
            L1=L.Linear(n_in, 100),
            L2=L.Linear(100, 200),
            L3=L.Linear(200, 100),
            L4=L.Linear(100, 100),
            q_value=L.Linear(100, n_out, initialW=np.zeros((n_out, 100), dtype=np.float32))
        )

    def q_func(self, in_layer):
        """
        Q function
        """
        layer1 = F.leaky_relu(self.L1(in_layer))
        layer2 = F.leaky_relu(self.L2(layer1))
        layer3 = F.leaky_relu(self.L3(layer2))
        layer4 = F.leaky_relu(self.L4(layer3))
        return self.q_value(layer4)

class Agent():
    def __init__(self, n_state, n_action, seed):
        np.random.seed(seed)
        sys.setrecursionlimit(10000)
        self.n_action = n_action
        self.model = Network(n_state, n_action)
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        self.memory = deque()
        self.loss = 0
        self.step = 0
        self.train_freq = 10
        self.target_update_freq = 20

        self.gamma = 0.99
        self.mem_size = 1000
        self.replay_size = 100
        self.epsilon = 0.05

    def stock_experience(self, exp):
        """
        経験をストックする
        """
        self.memory.append(exp)
        if len(self.memory) > self.mem_size:
            self.memory.popleft()

    def forward(self, exp):
        """
        順伝搬する
        """
        state = Variable(exp["state"])
        state_dash = Variable(exp["state_dash"])
        q_action = self.model.q_func(state)

        # Tartget
        tmp = self.target_model.q_func(state_dash)
        tmp = list(map(np.max, tmp.data))
        max_q_dash = np.asanyarray(tmp, dtype=np.float32)
        target = np.asanyarray(copy.deepcopy(q_action.data), dtype=np.float32)

        #  print "self.replay_size : ", self.replay_size
        for i in range(self.replay_size):
            #  print "self.replay_size : ", self.replay_size
            #  print "reward[", i, "] : ", exp["reward"][i]
            #  print "action [", i, "] : ", exp["action"][i]
            #  print "max_q_dash[", i, "] : ", max_q_dash[i]
            #  print "exp[epend][", i, "] : ", exp["ep_end"][i]
            target[i, exp["action"][i]] = exp["reward"][i] \
                + (self.gamma * max_q_dash[i]) * (not exp["ep_end"][i])
        loss = F.mean_squared_error(q_action, Variable(target))
        self.loss = loss.data

        """
        Clipping
        # CartPoleとかCartみたいに途中まででもうまくいかないと
        # 悪化しかしないものにClippingは向いていないのでは？
        # TODO: atariゲームでうまくいくかどうか確認したい
        ""
        for i in range(self.replay_size):
            if ep_end[i] is True:
                tmp_ = np.sign(reward[i])
            else:
                tmp_ = np.sign(reward[i]) + self.gamma * max_q_dash[i]
            target[i, action[i]] = tmp_

        # Clipping
        td = Variable(target) - Q
        td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)
        td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data)>1)

        zero_val = Variable(np.zeros((self.replay_size, self.n_action), dtype=np.float32))
        loss = F.mean_squared_error(Q, Variable(target))
        self.loss = loss.data
        """
        return loss

    def action(self, state):
        """
        状態を引数としてとり、
        Actionを選択｜生成して返す
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_action)
        else:
            state = Variable(state)
            q_action = self.model.q_func(state)
            q_action = q_action.data[0]
            act = np.argmax(q_action)
            return np.asarray(act, dtype=np.int8)

    def experience_replay(self):
        mem = np.random.permutation(np.array(self.memory))
        perm = np.array([i for i in range(len(mem))])
        for start in perm[::self.replay_size]:
            index = perm[start:start+self.replay_size]
            replay = mem[index]

            state = np.array([replay[i]["state"] \
                for i in range(self.replay_size)], dtype=np.float32)
            action = np.array([replay[i]["action"] \
                for i in range(self.replay_size)], dtype=np.int8)
            reward = np.array([replay[i]["reward"] \
                for i in range(self.replay_size)], dtype=np.float32)
            state_dash = np.array([replay[i]["state_dash"] \
                for i in range(self.replay_size)], dtype=np.float32)
            ep_end = np.array([replay[i]["ep_end"] \
                for i in range(self.replay_size)], dtype=np.bool)
            experience = {"state":state, "action":action, \
                "reward":reward, "state_dash":state_dash, "ep_end":ep_end}

            # 最適化
            self.model.zerograds()
            loss = self.forward(experience)
            loss.backward()
            self.optimizer.update()

    def train(self):
        if len(self.memory) >= self.mem_size:
            if self.step % self.train_freq == 0:
                self.experience_replay()
            if self.step % self.target_update_freq == 0:
                self.target_model = copy.deepcopy(self.model)
        self.step += 1

    def save_model(self, model_dir, env_name):
        serializers.save_npz(model_dir + "EnsekiTT_" + env_name + "_model.npz", self.model)

    def load_model(self, model_dir):
        serializers.load_npz(model_dir + "model.npz", self.model)
        self.target_model = copy.deepcopy(self.model)

def arg_parse():
    parser = argparse.ArgumentParser(description='いっけーAgent！n万エピソードだ！')
    parser.add_argument('--env', '-e', type=str, default="CartPole-v0",
                        help='Open AI environment')
    args = parser.parse_args()
    return args

def main():
    """
    Main Loop
    """
    args = arg_parse()
    # 所望のGymを作成する
    env = gym.make(args.env)

    # Gymの環境要素を取得する（State, Action，Seed(再現性のため)）
    n_state = env.observation_space.shape[0]
    #  CartPole-v0
    #  n_action = env.action_space.n

    #  Pendulum-v0
    action_list = [np.array([a]) for a in [-2.0, 2.0]]
    n_action = len(action_list)

    seed = 114514
    # Agentを作成する
    agent = Agent(n_state, n_action, seed)
    #  action_list = [i for i in range(0, n_action)]

    # 人がGymを見るためのモニタを開始する
    view_path = "/home/amsl/my_reinforcement_learning_tutorial/videos/EnsekiTT_dqn_" + args.env
    env = gym.wrappers.Monitor(env, view_path, video_callable=None, force=True)

    # エピソード回数分のループ
    for _episode in range(10000):
        # Gymの環境をリセットする
        observation = env.reset()
        # 時間数分のループ
        reward_sum = 0.0
        for _times in range(2000):
            # 環境をレンダリング
            env.render()
            # 状態を観察
            state = observation.astype(np.float32).reshape((1, n_state))
            # Agentは行動を選択する
            # CartPole-v0
            #  action = action_list[agent.action(state)]

            # Pendulum-v0
            action_i = agent.action(state)
            action = action_list[action_i]

            #  print "action_i : ", action_i
            # 環境で上記の行動を実効
            observation, reward, ep_end, _ = env.step(action)
            reward_sum += reward
            state_dash = observation.astype(np.float32).reshape((1, n_state))

            # CartPole-v0
            experience = {"state":state, "action":action, \
                "reward":reward, "state_dash":state_dash, "ep_end":ep_end}
            
            # Pendulum-v0
            experience = {"state":state, "action":action_i, \
                "reward":reward, "state_dash":state_dash, "ep_end":ep_end}

            # Agentは実行結果を経験としてストックする
            agent.stock_experience(experience)
            # Agentは学習する
            agent.train()
            # もし環境が終了判定をしていたらループを抜ける
            if ep_end:
                break
        print "episode : {0}, epsilon : {1}, reward sum : {2}, step : {3}".format(_episode, agent.epsilon, reward_sum, _times)
        agent.save_model('/home/amsl/my_reinforcement_learning_tutorial/models/deep_q_learning/', args.env)

if __name__ == "__main__":
    main()
