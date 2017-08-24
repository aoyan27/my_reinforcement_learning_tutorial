#!/usr/bin/env python
#coding:utf-8

import argparse

import gym
import numpy as np

from agents.pendulum_agent_trtd56_san import Agent

parser = argparse.ArgumentParser(description='Chainer example: test')
parser.add_argument('--env', '-e', default='Pendulum-v0', type=str,
                    help='Environment name')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()



def main(env_name, gpu, render=False, monitor=True, load=False, evaluation=False, seed=0):
	env = gym.make(env_name)

        video_path = "/home/amsl/my_reinforcement_learning_tutorial/videos/dqn_" + env_name
        model_path = "/home/amsl/my_reinforcement_learning_tutorial/models/" + env_name + "_"

        env = gym.wrappers.Monitor(env, video_path, force=True)

        n_st = env.observation_space.shape[0]

        max_episode = 2000
        
        #  Pendulum-v0
        action_list = [np.array([a]) for a in [-2.0, 2.0]]
        n_act = len(action_list)
        max_step = 200

        #  Acrobot-v1
        """
        n_act = env.action_space.n
        max_step = 500
        """

        agent = Agent(n_st, n_act, gpu, seed)

        if load:
            agent.load_model(model_path)
        
        count = 0

        for i_episode in xrange(max_episode):
            observation = env.reset()
            r_sum = 0.0
            q_list = []
            for t in xrange(max_step):
                if render:
                    env.render()

                state = observation.astype(np.float32).reshape((1, n_st))

                #  Pendulum-v0
                act_i, q = agent.get_action(state, evaluation)
                action = action_list[act_i]
                

                #  Acrobot-v1
                """
                action, q = agent.get_action(state, evaluation)
                """

                q_list.append(q)

                observation, reward, ep_end, _ = env.step(action)
                state_dash = observation.astype(np.float32).reshape((1, n_st))

                if not evaluation:
                    #  print "Now learning!!!!"
                    #  Peundulum-v0
                    agent.stock_experience(count, state, act_i, reward, state_dash, ep_end)

                    #  Acrobot-v1
                    """
                    agent.stock_experience(count, state, action, reward, state_dash, ep_end)
                    """

                    agent.train(count)

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


if __name__ == "__main__":
    main(args.env, args.gpu)

