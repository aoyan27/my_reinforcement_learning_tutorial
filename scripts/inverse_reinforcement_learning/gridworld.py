#!/usr/bin.env python
#coding:utf-8

import numpy as np

class Gridworld:
    def __init__(self, rows, cols, R_max, noise):
        self.rows = rows
        self.cols = cols
        self.n_state = self.rows * self.cols

        self.R_max = R_max

        self.noise = noise

        self.grid = np.zeros((self.rows, self.cols))
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
        self.grid[self.goal] = self.R_max

        self.action_list = [0, 1, 2, 3, 4]
        self.n_action = len(self.action_list)
        self.dirs = {0: '>', 1: '<', 2: 'v', 3: '^', 4: '-'}

        self._state = None

    def state2index(self, state):
        #  state[0] : x
        #  state[1] : y
        return state[1] + self.cols * state[0]

    def index2state(self, index):
        state = [0, 0]
        state[1] = index % self.cols
        state[0] = index / self.cols
        return state

    def get_next_state_and_probs(self, state, action):
        transition_probability = 1 - self.noise
        probs = np.zeros([self.n_action])
        probs[action] = transition_probability
        probs += self.noise / self.n_action
        #  print "probs : "
        #  print probs
        next_state_list = []

        for a in xrange(self.n_action):
            if state != list(self.goal):
                next_state, out_of_range = self.move(state, a)
                #  print "next_state() : "
                #  print next_state
                next_state_list.append(next_state)

                if out_of_range:
                    probs[self.n_action-1] += probs[a]
                    probs[a] = 0
            else:
                next_state = state
                #  print "probs[", a, "] : ", probs[a]
                if a != self.n_action-1:
                    probs[self.n_action-1] += probs[a]
                    probs[a] = 0
                next_state_list.append(next_state)
                #  print "next_state_ : "
                #  print next_state

        #  print "next_state_list : "
        #  print next_state_list
            
        #  print "probs_ : "
        #  print probs

        return next_state_list, probs

    def move(self, state, action):
        x, y = state
        if action == 0:
            #  right
            x = x + 1
        elif action == 1:
            #  left
            x = x - 1
        elif action == 2:
            #  down
            y = y + 1
        elif action == 3:
            #  up
            y = y - 1
        else:
            #  stay
            x = x
            y = y
        
        out_of_range = False
        if x < 0:
            x = 0
            out_of_range = True
        elif x > (self.cols-1):
            x = self.cols - 1
            out_of_range = True

        if y < 0:
            y = 0
            out_of_range = True
        elif y > (self.rows-1):
            y = self.rows - 1
            out_of_range = True

        return [x, y], out_of_range
    
    def show_policy(self, policy):
        vis_policy = np.array([])
        for i in xrange(len(policy)):
            vis_policy = np.append(vis_policy, self.dirs[policy[i]])
            #  print self.dirs[policy[i]]
        vis_policy = vis_policy.reshape((self.rows, self.cols))
        vis_policy[self.goal] = 'G'
        print vis_policy


    def reset(self):
        self._state = [0, 0]
        return self._state

    def step(self, action, reward_map=None):
        next_state_list, probs = self.get_next_state_and_probs(self._state, action)
        #  print "next_state_list : ", next_state_list
        #  print "probs : ", probs
        random_num = np.random.rand()
        #  print "random_num : ", random_num
        action_index = 0
        for i in xrange(len(probs)):
            random_num -= probs[i]
            #  print "random_num_ : ", random_num
            if random_num < 0:
                action_index = i
                break
        #  print "action_index : ", action_index
        #  print "next_state : ", next_state_list[action_index]

        self._state = next_state_list[action_index]
        #  self._state, _ = self.move(self._state, action)
        

        reward = None
        if reward_map is None:
            if self._state == list(self.goal):
                reward = self.R_max
            else:
                reward = 0
        else:
            reward = reward_map[self.state2index(self._state)]
            #  print "reward : ", reward

        episode_end = False
        if self._state == list(self.goal):
            episode_end = True

        return self._state, reward, episode_end, {'probs':probs, 'random_num':random_num}


if __name__=="__main__":
    rows = 5
    cols = 5
    R_max = 10.0

    noise = 0.3

    env = Gridworld(rows, cols, R_max, noise)

    print "env.n_state : ", env.n_state
    print "env.n_action : ", env.n_action
    
    max_episode = 1000
    max_step = 200

    reward_map = np.load('./reward_map.npy')
    print "reward_map : "
    print  reward_map

    for i in xrange(max_episode):
        print "================================================="
        print "episode : ", i+1
        observation = env.reset()
        for j in xrange(max_step):
            print "---------------------------------------------"
            state = observation
            print "state : ", state

            action = np.random.randint(env.n_action)
            print "action : ", action, env.dirs[action]

            #  observation, reward, done, info = env.step(action)
            observation, reward, done, info = env.step(action, reward_map)
            next_state = observation

            print "next_state : ", next_state
            print "reward : ", reward
            print "episode_end : ", done
            #  print "info : ", info
            print "step : ", j+1

            if done:
                break

            


