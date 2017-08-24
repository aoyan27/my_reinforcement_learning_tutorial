#!/usr/bin/env python
#coding:utf-8

import gym, sys
import numpy as np

from agents.agent import Agent

def main(env_name, render=False, monitor=True, load=False, evaluation=False, seed=0):
	env = gym.make(env_name)

        video_path = "/home/amsl/my_reinforcement_learning_tutorial/videos/orig_dqn_" + env_name
        model_path = "/home/amsl/my_reinforcement_learning_tutorial/models/orig_" + env_name + "_"

        env = gym.wrappers.Monitor(env, video_path, force=True)

        n_st = env.observation_space.shape[0]
        
        #  For Pendulum-v0
        action_list = [np.array([a]) for a in [-2.0, 2.0]]
        n_act = len(action_list)

        #  For Acrobot-v1
        """
        n_act = env.action_space.n
        """

        agent = Agent(n_st, n_act, seed)

        if load:
            agent.load_model(model_path)
        
        count = 0

        for i_episode in xrange(5000):
            observation = env.reset()
            r_sum = 0.0
            q_list = []
            for t in xrange(500):
                if render:
                    env.render()

                state = observation.astype(np.float32).reshape((1, n_st))

                #  For Pendulum-v0
                act_i, q = agent.get_action(state, evaluation)
                #  print "type(act_i) : ", type(act_i)
                #  act_i = 0
                #  q = 0.0
                q_list.append(q)
                action = action_list[act_i]

                #  For Acrobot-v1
                """
                action, q = agent.get_action(state, evaluation)
                q_list.append(q)
                """
                observation, reward, ep_end, _ = env.step(action)
                state_dash = observation.astype(np.float32).reshape((1, n_st))
                if not evaluation:
                    #  print "Now learning!!!!"
                    #  For Pendulum-v0
                    agent.stock_experience(count, state, act_i, reward, state_dash, ep_end)

                    #  For Acrobot-v1
                    """
                    agent.stock_experience(count, state, action, reward, state_dash, ep_end)
                    """

                    agent.train(count)
                    #  pass
                r_sum += reward
                #  print "count : ", count % 10
                #  print "state : ", state
                #  print "act_i : ", act_i
                #  print "reward : ", reward
                #  print "state_dash : ", state_dash
                #  print "ep_end : ", ep_end
                
                count += 1
                
                if ep_end:
                    break
            print "Episode : %d\t/Reward Sum : %f\t/Epsilon : %f\t/Loss : %f\t/Average Q : %f\t/Time Step : %d" % (i_episode, r_sum, agent.epsilon, agent.loss, sum(q_list)/float(t+1), agent.step)
            #  print "\t".join(map(str, [i_episode, r_sum, agent.epsilon, agent.loss, sum(q_list)/float(t+1), agent.step]))
            if not evaluation:
                agent.save_model(model_path)
                #  pass

if __name__=="__main__":
    env_name = sys.argv[1]
    main(env_name)
    #  main(env_name, render=True,  monitor=False, load=True, evaluation=True)


