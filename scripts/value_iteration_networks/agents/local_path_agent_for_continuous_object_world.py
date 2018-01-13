#!/usr/bin/env python
#coding:utf-8

import numpy as np
import copy
import math

class LocalPlanAgent:
    def __init__(self, env, global_planner):
        self.env = env
        self.global_planner = global_planner

        self.evaluation_time = 0.5
        self.dt = 0.05

        self.future_traj_position_list = [[] for i in xrange(self.env.n_continuous_action)]
        self.future_traj_orientation_list = [[] for i in xrange(self.env.n_continuous_action)]
        self.future_traj_collision_list = [[] for i in xrange(self.env.n_continuous_action)]
        self.future_traj_out_of_range_list = [[] for i in xrange(self.env.n_continuous_action)]

        self.continuous_global_path_list = None

        self.evaluation_value = [0 for i in xrange(self.env.n_continuous_action)]

        self.selected_traj_position_list = None
        self.selected_traj_orientation_list = None

    def get_future_trajectory(self, position, orientation):
        for i_action in self.env.continuous_action_list:
            #  print "i_action : ", i_action
            y, x = position
            yaw = orientation

            self.future_traj_position_list[i_action].append(position)
            self.future_traj_orientation_list[i_action].append(orientation)
            self.future_traj_out_of_range_list[i_action].append(False)
            self.future_traj_collision_list[i_action].append(False)


            t = 0.0
            while t <= self.evaluation_time:
                #  print "t : ", t
                next_position, next_orientation, out_of_range, collision \
                        = self.env.continuous_move([y, x], yaw, i_action)

                self.future_traj_position_list[i_action].append(next_position)
                self.future_traj_orientation_list[i_action].append(next_orientation)
                self.future_traj_out_of_range_list[i_action].append(out_of_range)
                self.future_traj_collision_list[i_action].append(collision)
                y, x = next_position
                yaw = next_orientation
                if out_of_range or collision:
                    #  print "out_of_range or collision !!!"
                    break
                t += self.dt
        #  print "self.future_traj_position_list : "
        #  print self.future_traj_position_list
        #  self.env.show_continuous_objectworld(local_path=self.future_traj_position_list)

    def transform_global_path_discreate2continuous(self, global_state_list):
        tmp_global_state = np.asarray(copy.deepcopy(global_state_list)).transpose(1, 0)
        #  print "tmp_global_state : ", tmp_global_state
        global_path_continuous_y, global_path_continuous_x \
                = self.env.discreate2continuous(tmp_global_state[0], tmp_global_state[1])
        global_path_continuous_y += self.env.cell_size / 2.0
        global_path_continuous_x += self.env.cell_size / 2.0
        self.continuous_global_path_list \
                = np.array([global_path_continuous_y, global_path_continuous_x]).transpose(1, 0)
        #  print "self.continuous_global_path_list : "
        #  print self.continuous_global_path_list 
        #  self.env.show_continuous_objectworld(global_path=self.continuous_global_path_list)
    
    def calc_dist(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def calc_dist_line_and_point(self, a, b, p):
        u = np.array([b[0]-a[0], b[1]-a[1]])
        v = np.array([p[0]-a[0], p[1]-a[1]])
        L = abs(np.cross(u, v) / np.linalg.norm(u))
        return L
    
    def check_inverse_global_path_distance(self, final_position, global_state_list):
        # global_pathとの距離の逆数を評価
        #  max_range_y = np.max([global_state_list[0][0], global_state_list[-1][0]])
        #  max_range_x = np.max([global_state_list[0][1], global_state_list[-1][1]])
        #  min_range_y = np.min([global_state_list[0][0], global_state_list[-1][0]])
        #  min_range_x = np.min([global_state_list[0][1], global_state_list[-1][1]])
        #  if [min_range_y, min_range_x] <= final_position \
                #  and final_position <= [max_range_y, max_range_x]:
        if len(global_state_list) < 2:
            L = self.calc_dist(final_position, global_state_list[0])
        else:
            tmp_global_dist_list = []
            for global_position in global_state_list:
                tmp_global_dist_list.append(self.calc_dist(final_position, global_position))

            #  print "tmp_global_dist_list : ", tmp_global_dist_list
            min_index = np.argsort(np.asarray(tmp_global_dist_list))
            #  print "min_index : ", min_index
            
            global_position_1 = global_state_list[min_index[0]]
            global_position_2 = global_state_list[min_index[1]]
            #  print "global_position_1 : ", global_position_1
            #  print "global_position_2 : ", global_position_2
            L = self.calc_dist_line_and_point(global_position_1, global_position_2, final_position)
            #  print "L : ", L
        if L == 0.0:
            L = 0.0001

        inverse_global_path_distance = 1.0 / L
        #  else:
            #  inverse_global_path_distance = 0.0

        return inverse_global_path_distance

    def check_nearest_obstacle_distance(self, final_position):
        #  最近傍の障害物との距離
        tmp_obs_dist_list = []
        for obstacle in self.env.continuous_objects:
            tmp_obs_dist_list.append(self.calc_dist(final_position, obstacle))
        #  print "tmp_obs_dist_list : ", tmp_obs_dist_list
        max_obs_dist = 0
        if len(tmp_obs_dist_list) != 0:
            max_obs_dist = np.max(tmp_obs_dist_list)
        #  print "max_obs_dist : ", max_obs_dist

        return max_obs_dist

    def check_diffarence_goal_heading(self, final_position, final_orientation):
        #  ゴールと自身の方位の差
        #  print "final_orientation : ", math.degrees(final_orientation)
        goal_orientation = math.atan2(self.env.goal[0]-final_position[0], \
                                      self.env.goal[1]-final_position[1])

        #  print "goal_orientation : ", math.degrees(goal_orientation)
        diffarence_goal_heading = np.pi -  math.fabs(goal_orientation - final_orientation)
        #  print "diffarence_goal_heading : ", math.degrees(diffarence_goal_heading)
        
        return diffarence_goal_heading

    def vector_normalize(self, input_vector):
        input_vector = np.asarray(input_vector)
        normal_vector = copy.deepcopy(input_vector)
        norm = np.linalg.norm(input_vector)
        if norm != 0:
            normal_vector = input_vector / norm

        return normal_vector


    def evaluation_local_path(self, local_position_list, local_orientation_list, global_state_list):
        #  print "local_position_list : ", local_position_list
        #  print "local_orientaiton_list : ", local_orientation_list
        self.evaluation_value = [0 for i in xrange(self.env.n_continuous_action)]

        inverse_global_path_distance = [0 for i in xrange(self.env.n_continuous_action)]
        nearest_obstacle_distance = [0 for i in xrange(self.env.n_continuous_action)]
        diffarence_goal_heading = [0 for i in xrange(self.env.n_continuous_action)]
        velocity_list = [0 for i in xrange(self.env.n_continuous_action)]

        
        for i_action in self.env.continuous_action_list:
            #  if  not self.future_traj_out_of_range_list[i_action][-1] \
                    #  and  not self.future_traj_collision_list[i_action][-1]:
            if not self.future_traj_collision_list[i_action][-1]:

                final_position = local_position_list[i_action][-1]

                # global_pathとの距離の逆数を評価
                inverse_global_path_distance[i_action] \
                        = self.check_inverse_global_path_distance(final_position, global_state_list)

                #  最近傍の障害物との距離
                nearest_obstacle_distance[i_action] \
                        = self.check_nearest_obstacle_distance(final_position)

                #  ゴールと自身の方位の差
                final_orientation = local_orientation_list[i_action][-1]
                diffarence_goal_heading[i_action] \
                        = self.check_diffarence_goal_heading(final_position, final_orientation)

            #  自身の速度
            velocity_list[i_action] = self.env.velocity_vector[i_action][0]


        #  print "inverse_global_path_distance : ", inverse_global_path_distance
        #  print "nearest_obstacle_distance : ", nearest_obstacle_distance
        #  print "diffarence_goal_heading : ", diffarence_goal_heading
        #  print "velocity_list : ", velocity_list
        
        a = 10.0
        b = 20.0
        c = 5.0
        d = 1.0

        self.evaluation_value = a * self.vector_normalize(inverse_global_path_distance) \
                              + b * self.vector_normalize(nearest_obstacle_distance) \
                              + c * self.vector_normalize(diffarence_goal_heading) \
                              + d * self.vector_normalize(velocity_list)
        #  print "self.evaluation_value : "
        #  print self.evaluation_value
        optimal_action = np.argmax(self.evaluation_value)
        #  print "optimal_action : ", optimal_action
        self.selected_traj_position_list = self.future_traj_position_list[optimal_action]
        self.selected_traj_orientation_list = self.future_traj_orientation_list[optimal_action]
        #  self.env.show_continuous_objectworld(selected_path=self.selected_traj_position_list)
        return optimal_action

                

if __name__ == "__main__":
    import sys
    sys.path.append('../')
    from envs.continuous_state_object_world import Objectworld
    from a_star_agent_for_continuous_object_world import AstarAgent


    height = 10
    width = 10
    cell_size = 0.25
    rows = int(height / cell_size)
    cols = int(width / cell_size)
    goal = [height-1, width-1]

    R_max = 1.0
    noise = 0.0
    n_objects = 100
    seed = 1

    env = Objectworld(rows, cols, cell_size, goal, R_max, noise, n_objects, seed, mode=1)

    #  print "env.grid : "
    #  env.show_objectworld()

    i = 0

    #  a_agent = AstarAgent(env)

    #  start_position = [0, 0]
    #  a_agent.a_star(start_position)
    success_times = 0
    failed_times = 0
    while i < 5:
        print "i : ", i
        #  env.set_start_random()
        #  print "env.state_ : ", env.state_
        #  env.set_goal_random()
        #  env.set_objects()
        #  print "env.grid : "
        #  env.show_objectworld()
        env.reset(random=True, goal_heading=True)
        print "env.state_ : ", env.state_
        print "env.orientation_ : ", env.orientation_
        #  env.set_objects()
        env.calc_continuous_trajectory()


        a_agent = AstarAgent(env)
        start_position = list(env.continuous2discreate(env.state_[0], env.state_[1]))
        print "start_position : ", start_position
        a_agent.get_shortest_path(start_position)
        path_data = None
        if a_agent.found:
            path_data = a_agent.show_path()
            print "view_path : "
            a_agent.view_path(path_data['vis_path'])

        
            for i_step in xrange(10000):
                print "=============================================="
                print "step : ", i_step

                l_agent = LocalPlanAgent(env, a_agent)

                #  start_position = list(env.continuous2discreate(env.state_[0], env.state_[1]))
                #  print "start_position : ", start_position
                #  a_agent.a_star(start_position)

                #  a_agent.get_shortest_path(start_position)
                #  print "a_agent.expand_list : "
                #  print a_agent.expand_list
                #  print "a_agent.action_list : "
                #  print a_agent.action_list
                
                if a_agent.found:
                    #  print "a_agent.state_list : "
                    #  print a_agent.state_list
                    #  print "a_agent.shrotest_action_list : "
                    #  print a_agent.shortest_action_list
                    #  env.show_policy(a_agent.policy.transpose().reshape(-1))
                    #  path_data = a_agent.show_path()
                    #  print "view_path : "
                    #  a_agent.view_path(path_data['vis_path'])
                    #  print "state_list : ", list(path_data['state_list'])
                    #  print "action_list : ", path_data['action_list']
                    
                    l_agent.transform_global_path_discreate2continuous(path_data['state_list'])

                    l_agent.get_future_trajectory(env.state_, env.orientation_)
                    continuous_action \
                            = l_agent.evaluation_local_path(l_agent.future_traj_position_list, \
                            l_agent.future_traj_orientation_list, l_agent.continuous_global_path_list)
                    print "l_agent.evaluation_value : ", l_agent.evaluation_value
                    print "continuous_action : ", continuous_action, env.velocity_vector[continuous_action]
                    #  continuous_action = env.get_action_sample(continuous=True)

                    #  env.show_continuous_objectworld(global_path=l_agent.continuous_global_path_list)
                    env.show_continuous_objectworld(global_path=l_agent.continuous_global_path_list, \
                            local_path=l_agent.future_traj_position_list, \
                            selected_path=l_agent.selected_traj_position_list)


                    next_state, orientation, reward, done, info = env.step(continuous_action)
                    print "reward : ", reward
                    print "episode_end : ", done
                    print info

                    if done:
                        if reward > 0:
                            success_times += 1
                        if reward <= 0:
                            failed_times += 1

                        i += 1
                        break


        print "Success_times : ", success_times
        print "Failed_times : ", failed_times
                
