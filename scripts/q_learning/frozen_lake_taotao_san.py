#!/usr/bin/env python3
#config:utf-8


import sys
import os.path
import pprint

import numpy as np
import gym


DEBUG = False
#DEBUG = True

ENVS = {
    "4x4" : "FrozenLake-v0",
    "8x8" : "FrozenLake8x8-v0",
}


def error(msg):
    sys.exit(msg)


class Agent:
    def __init__(self, env):
        self.env = env
        self.q = [[.0,.0,.0,.0] for _ in range(env.observation_space.n)]

    def learn(self, alpha, gamma):
        state = self.env.reset()
        if DEBUG: self.env.render()

        for t in range(self.env.spec.timestep_limit):
            act = self.env.action_space.sample()
            state_next, reward, done, info = self.env.step(act)

            q_next_max = max(self.q[state_next])
            self.q[state][act] = (1-alpha) * self.q[state][act]\
                                 + alpha * (reward + gamma*q_next_max)

            if DEBUG:
                self.env.render()
                print(state_next, reward, done, info)
                pprint.pprint(self.q)

            if done:
                return reward
            else:
                state = state_next
        return 0.0

    def test(self):
        state = self.env.reset()
        if DEBUG: self.env.render()

        for t in range(self.env.spec.timestep_limit):
            act = np.argmax(self.q[state])
            state, reward, done, info = self.env.step(act)

            if DEBUG:
                self.env.render()
                print(state, reward, done, info)

            if done:
                return reward
        return 0.0


def usage():
    error("Usage: FrozenLake-qlearning <4x4|8x8> <alpha> <gamma> <learn_count> <test_count> [recdir]")

def main():
    if len(sys.argv) < 6: usage()
    env_name = ENVS[sys.argv[1]]
    alpha = float(sys.argv[2])
    gamma = float(sys.argv[3])
    learn_count = int(sys.argv[4])
    test_count = int(sys.argv[5])
    rec_dir = sys.argv[6] if len(sys.argv) >= 7 else None
    print("# <{}> alpha={}, gamma={}, learn_count={} test_count={}".format(
        env_name, alpha, gamma, learn_count, test_count))

    env = gym.make(env_name)
    print("# step-max: {}".format(env.spec.timestep_limit))
    if rec_dir:
        subdir = "FrozenLake{}-qlearning-alpha{}-gamma{}-learn{}-test{}".format(
            sys.argv[1], alpha, gamma, learn_count, test_count
        )
        env.monitor.start(os.path.join(rec_dir, subdir))
    agent = Agent(env)

    print("##### LEARNING #####")
    reward_total = 0.0
    for episode in range(learn_count):
        reward_total += agent.learn(alpha, gamma)
    pprint.pprint(agent.q)
    print("episodes:       {}".format(learn_count))
    print("total reward:   {}".format(reward_total))
    print("average reward: {:.3f}".format(reward_total / learn_count))

    print("##### TEST #####")
    reward_total = 0.0
    for episode in range(test_count):
        reward_total += agent.test()
    print("episodes:       {}".format(test_count))
    print("total reward:   {}".format(reward_total))
    print("average reward: {:.3f}".format(reward_total / test_count))

    if rec_dir: env.monitor.close()

if __name__ == "__main__": main()
