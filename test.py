"""
retrieved from https://github.com/princewen/tensorflow_practice/blob/master/RL/Basic-MADDPG-Demo/three_agent_maddpg.py
modified the code to make it work with our environment
test the env
"""

import numpy as np
import tensorflow as tf
from env_FireFighter import EnvFireFighter
from maddpg_agent import MADDPG
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
from collections import deque
import os

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def create_init_update(online_name, target_name, tau=0.99):
    """
    This creates an init operation, which will initialize the weights of both online and target models of an agent to be
    equal. It also creates the update operation, which when called, will adjust the weights of the target model to be closer
    to that of the online model.
    :param online_name: The scope of the online model, so we can retrieve its trainable variables (weights)
    :param target_name: The scope of the target model
    :param tau: A parameter which specifies how much to adjust the target weights in the direction of the online weights
    :return:
    """

    online_var = [i for i in tf.trainable_variables() if online_name in i.name]

    target_var = [i for i in tf.trainable_variables() if target_name in i.name]


    target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]

    target_update = [tf.assign(target, (1 - tau) * online + tau * target) for online, target in zip(online_var, target_var)]



    return target_init, target_update







def get_agents_action(obs, sess, noise_rate=0):

    # debugged here
    obs = np.asarray(obs)
    o_1 = np.expand_dims(np.expand_dims(obs[0],axis=0),axis=0)
    o_2 = np.expand_dims(np.expand_dims(obs[1],axis=0),axis=0)

    act1 = agent1_ddpg.action(state=o_1, sess=sess)

    act2 = agent2_ddpg.action(state=o_2, sess=sess)


    return act1, act2


def train_agent(agent_ddpg, agent_ddpg_target, agent_memory, agent_actor_target_update, agent_critic_target_update, sess, other_actors, num_agents=2):
    """
    This is an important function, which runs a single train step for a single agent.
    :param agent_ddpg: The online part of agent which we will be training. This is the object which represents the deep network that we choose actions from
    :param agent_ddpg_target: The target part of the agent. We base our update values on this agent's output
    :param agent_memory: This is the agent's memory. It includes up to 2000 tuples of (all agents' obs, all agents' actions, agent's reward, all agents' next obs, done)
    :param agent_actor_target_update: The update operation for the actor network. Will update actor network's target weights to be more equal to the online weights
    :param agent_critic_target_update: The update operation for the critic network.
    :param sess: Session object to run tensorflow operations
    :param other_actors: A list of target models for the other agents in the environment.
    :return:
    """


    total_obs_batch, total_act_batch, rew_batch, total_next_obs_batch, done_mask = agent_memory.sample(32)

    act_batch = total_act_batch[:, 0, :]

    for i in range(num_agents-1):
        other_act = total_act_batch[:, i+1, :]
        if i == 0:
            other_act_batch = other_act
        else:
            other_act_batch = np.hstack([other_act_batch, other_act])
        #other_act_batch = np.hstack([total_act_batch[:, 1, :], total_act_batch[:, 2, :]])



    obs_batch = total_obs_batch[:, 0, :]



    next_obs_batch = total_next_obs_batch[:, 0, :]

    next_other_actor1_o = total_next_obs_batch[:, 1, :]


    # 获取下一个情况下另外两个agent的行动
    next_other_action = other_actors[0].action(next_other_actor1_o, sess)

    #next_other_action = np.hstack([other_actors[0].action(next_other_actor1_o, sess), other_actors[1].action(next_other_actor2_o, sess)])

    target = rew_batch.reshape(-1, 1) + 0.9999 * agent_ddpg_target.Q(state=next_obs_batch, action=agent_ddpg.action(next_obs_batch, sess),

                                                                     other_action=next_other_action, sess=sess)

    agent_ddpg.train_actor(state=obs_batch, other_action=other_act_batch, sess=sess)

    agent_ddpg.train_critic(state=obs_batch, action=act_batch, other_action=other_act_batch, target=target, sess=sess)



    sess.run([agent_actor_target_update, agent_critic_target_update])





if __name__ == '__main__':

    save_dir = "saves"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)








    agent1_ddpg = MADDPG('agent1')

    agent1_ddpg_target = MADDPG('agent1_target')

    agent2_ddpg = MADDPG('agent2')

    agent2_ddpg_target = MADDPG('agent2_target')

    saver = tf.train.Saver()

    agent1_actor_target_init, agent1_actor_target_update = create_init_update('agent1_actor', 'agent1_target_actor')

    agent1_critic_target_init, agent1_critic_target_update = create_init_update('agent1_critic', 'agent1_target_critic')

    agent2_actor_target_init, agent2_actor_target_update = create_init_update('agent2_actor', 'agent2_target_actor')

    agent2_critic_target_init, agent2_critic_target_update = create_init_update('agent2_critic', 'agent2_target_critic')


    # four houses and three agents
    num_houses = 3
    env = EnvFireFighter(num_houses)

    #o_n = env.get_obs()
    # observation is the state for each figherfighter ex: [[1,0],[1,1],[1,0]]





    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    sess.run([agent1_actor_target_init, agent1_critic_target_init,

              agent2_actor_target_init, agent2_critic_target_init])


    num_episodes = 10000

    rewards = []
    average_over = int(num_episodes / 10)
    average_rewards = []
    average_r = deque(maxlen=average_over)





    agent1_memory = ReplayBuffer(2000) #This was at 100

    agent2_memory = ReplayBuffer(2000)




    # e = 1





    batch_size = 32

    num_steps = 200


    for i in range(num_episodes):
    # make graph with reward

        #Reset the environment at the start of each episode
        o_n = env.reset()

        total_ep_reward = 0

        for t in range(num_steps):


            #Get action probabilities at each timestep
            agent1_action, agent2_action = get_agents_action(o_n, sess, noise_rate=0.2)



            #三个agent的行动
            agent1_action = np.squeeze(agent1_action)
            agent2_action = np.squeeze(agent2_action)

            #Sample from probabilities
            action1 = np.random.choice(np.arange(len(agent1_action)), p=agent1_action)
            action2 = np.random.choice(np.arange(len(agent2_action)), p=agent2_action)

            a = [action1, action2]
            #print("action of agent is " + str(a))
            # global reward as the reward for each agent

            #Get global reward
            glob_reward = env.step(a)
            r_n = [glob_reward for _ in range(2)]


            total_ep_reward += glob_reward

            #print("reward is " + str(r_n))
            #Get next state
            o_n_next = env.firelevel

            #Add to agents' memories the state, actions, reward, next state, done tuples
            agent1_memory.add(np.vstack([o_n[0], o_n[1]]), np.vstack([agent1_action, agent2_action]), r_n[0], np.vstack([o_n_next[0], o_n_next[1]]), False)



            agent2_memory.add(np.vstack([o_n[1], o_n[2], o_n[0]]), np.vstack([agent2_action,agent1_action]), r_n[1], np.vstack([o_n_next[1], o_n_next[0]]), False)




            # original is 50000
            if t > batch_size:

                # e *= 0.9999

                # agent1 train

                #Run a single train step for each agent
                train_agent(agent1_ddpg, agent1_ddpg_target, agent1_memory, agent1_actor_target_update,

                            agent1_critic_target_update, sess, [agent2_ddpg_target])



                train_agent(agent2_ddpg, agent2_ddpg_target, agent2_memory, agent2_actor_target_update,

                            agent2_critic_target_update, sess, [agent1_ddpg_target])


                #print("step " + str(t))
                #print("observation by agents are  " + str(env.get_obs()) + "\n\n")
                #print("fire level at step " + str(t) + " is " + str(env.firelevel))

            o_n = o_n_next

        print("Episode: {}. Global Reward: {}.".format(i+1, total_ep_reward))
        rewards.append(total_ep_reward)
        average_r.append(total_ep_reward)

        if i < average_over:
            r = 0
            for j in range(i):
                r += average_r[j]
            r /= (i + 1)
            average_rewards.append(r)
        else:
            average_rewards.append(sum(average_r) / average_over)

        if i % average_over == 0:
            plt.plot(rewards)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.savefig(save_dir + "/reward.png")
            plt.clf()

            plt.plot(average_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Moving average')
            plt.savefig(save_dir + "/moving_avg.png")
            plt.clf()


    sess.close()


