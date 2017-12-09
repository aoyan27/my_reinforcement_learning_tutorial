#!/usr/bin/env python
#coding:utf-8

import numpy as np
np.set_printoptions(suppress=True, threshold=np.inf)


class DijkstraAgent:
    def __init__(self, env):
        self.env = env

        self.cost = 1
        self.open_list = []
        self.closed_list = np.zeros([self.env.rows, self.env.cols])
        self.expand_list = np.empty([self.env.rows, self.env.cols])
        self.expand_list.fill(-1)
        self.action_list = np.empty([self.env.rows, self.env.cols])
        self.action_list.fill(-1)
        self.policy = None
        self.state_list = []
        self.shortest_action_list = []

        self.found = False
        self.resign = False
    
    def dijkstra(self, start_position):
        g = 0

        self.open_list.append([g, start_position[0], start_position[1]])
        self.closed_list[tuple(start_position)] = 1

        found = False
        resign = False

        n = 0
        
        while not found and not resign:
            if len(self.open_list) == 0:
                #  print "resign!!!"
                resign = True
                self.resign = resign
            else:
                #  print "self.open_list : ", self.open_list
                self.open_list.sort()
                self.open_list.reverse()
                current = self.open_list.pop()
                #  print "current : ", current
                g = current[0]
                state = [current[1], current[2]]
                #  print "state : ", state

                self.expand_list[tuple(state)] = n
                n += 1

                #  print "self.env.goal : ", self.env.goal
                if state == self.env.goal:
                    found = True
                    self.found = found

                for a in xrange(len(self.env.action_list)):
                    next_state, out_of_range, collision = self.env.move(state, a)
                    #  print "next_state : ", next_state
                    #  print "out_of_range : ", out_of_range
                    #  print "collision : ", collision
                    
                    if not out_of_range:
                        if not collision and self.closed_list[tuple(next_state)] == 0:
                            #  print "next_state(ok) : ", next_state
                            next_g = g + self.cost
                            self.open_list.append([next_g, next_state[0], next_state[1]])
                            self.closed_list[tuple(next_state)] = 1
                            # self.action_listは、その状態に最初に訪れるときに、
                            # 直前の状態において実行した行動が格納される
                            # (self.closed_listでその状態に訪問したかをカウントされているため、
                            # その状態への２回目の訪問はないから)
                            self.action_list[tuple(next_state)] = a
                            

    def get_shortest_path(self, start_position):
        self.dijkstra(start_position)
        
        if self.found:
            self.policy = np.empty([self.env.rows, self.env.cols])
            self.policy.fill(8)
            state = self.env.goal
            self.state_list.append(state)
            self.shortest_action_list.append(self.policy[tuple(state)])
            
            while state != start_position:
                before_state, _, _ = \
                        self.env.move(state, self.action_list[tuple(state)], reflect=-1)
                #  print "before_state : ", before_state
                self.policy[tuple(before_state)] = self.action_list[tuple(state)]
                self.state_list.append(before_state)
                self.shortest_action_list.append(self.policy[tuple(before_state)])
                state = before_state
            self.state_list.reverse()
            self.shortest_action_list.reverse()



if __name__ == "__main__":
    import sys
    sys.path.append('../')
    from envs.object_world import Objectworld
    rows = 5
    cols = 5
    goal = [rows-1, cols-1]

    R_max = 1.0
    noise = 0.0
    n_objects = 5
    seed = 1

    env = Objectworld(rows, cols, goal, R_max, noise, n_objects, seed, mode=1)

    #  print "env.grid : "
    #  env.show_objectworld()

    i = 0
    while i < 10:
        print "i : ", i
        env.set_objects()
        print "env.grid : "
        env.show_objectworld()

        d_agent = DijkstraAgent(env)

        start_position = [0, 0]
        #  d_agent.dijkstra(start_position)
        d_agent.get_shortest_path(start_position)
        #  print "d_agent.expand_list : "
        #  print d_agent.expand_list
        #  print "d_agent.action_list : "
        #  print d_agent.action_list
        
        if d_agent.found:
            print "d_agent.state_list : "
            print d_agent.state_list
            print "d_agent.shrotest_action_list : "
            print d_agent.shortest_action_list
            env.show_policy(d_agent.policy.transpose().reshape(-1))
            i += 1
