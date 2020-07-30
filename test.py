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

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def create_init_update(oneline_name, target_name, tau=0.99):

    online_var = [i for i in tf.trainable_variables() if oneline_name in i.name]

    target_var = [i for i in tf.trainable_variables() if target_name in i.name]


    target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]

    target_update = [tf.assign(target, (1 - tau) * online + tau * target) for online, target in zip(online_var, target_var)]



    return target_init, target_update



agent1_ddpg = MADDPG('agent1')

agent1_ddpg_target = MADDPG('agent1_target')


agent2_ddpg = MADDPG('agent2')

agent2_ddpg_target = MADDPG('agent2_target')


agent3_ddpg = MADDPG('agent3')

agent3_ddpg_target = MADDPG('agent3_target')

saver = tf.train.Saver()

agent1_actor_target_init, agent1_actor_target_update = create_init_update('agent1_actor', 'agent1_target_actor')

agent1_critic_target_init, agent1_critic_target_update = create_init_update('agent1_critic', 'agent1_target_critic')


agent2_actor_target_init, agent2_actor_target_update = create_init_update('agent2_actor', 'agent2_target_actor')

agent2_critic_target_init, agent2_critic_target_update = create_init_update('agent2_critic', 'agent2_target_critic')


agent3_actor_target_init, agent3_actor_target_update = create_init_update('agent3_actor', 'agent3_target_actor')

agent3_critic_target_init, agent3_critic_target_update = create_init_update('agent3_critic', 'agent3_target_critic')


def get_agents_action(o_n, sess, noise_rate=0):

    # debugged here
    o_n = np.asarray(o_n)
    o_1 = np.expand_dims(np.expand_dims(o_n[0],axis=0),axis=0)
    o_2 = np.expand_dims(np.expand_dims(o_n[1],axis=0),axis=0)
    o_3 = np.expand_dims(np.expand_dims(o_n[2],axis=0),axis=0)
    agent1_action = agent1_ddpg.action(state=o_1, sess=sess)

    agent2_action = agent2_ddpg.action(state=o_2, sess=sess)

    agent3_action = agent3_ddpg.action(state=o_3, sess=sess)

    return agent1_action, agent2_action, agent3_action


def train_agent(agent_ddpg, agent_ddpg_target, agent_memory, agent_actor_target_update, agent_critic_target_update, sess, other_actors):

    total_obs_batch, total_act_batch, rew_batch, total_next_obs_batch, done_mask = agent_memory.sample(32)



    act_batch = total_act_batch[:, 0, :]

    other_act_batch = np.hstack([total_act_batch[:, 1, :], total_act_batch[:, 2, :]])



    obs_batch = total_obs_batch[:, 0, :]



    next_obs_batch = total_next_obs_batch[:, 0, :]

    next_other_actor1_o = total_next_obs_batch[:, 1, :]

    next_other_actor2_o = total_next_obs_batch[:, 2, :]

    # 获取下一个情况下另外两个agent的行动

    next_other_action = np.hstack([other_actors[0].action(next_other_actor1_o, sess), other_actors[1].action(next_other_actor2_o, sess)])

    target = rew_batch.reshape(-1, 1) + 0.9999 * agent_ddpg_target.Q(state=next_obs_batch, action=agent_ddpg.action(next_obs_batch, sess),

                                                                     other_action=next_other_action, sess=sess)

    agent_ddpg.train_actor(state=obs_batch, other_action=other_act_batch, sess=sess)

    agent_ddpg.train_critic(state=obs_batch, action=act_batch, other_action=other_act_batch, target=target, sess=sess)



    sess.run([agent_actor_target_update, agent_critic_target_update])





if __name__ == '__main__':


    # four houses and three agents
    num_houses = 4
    env = EnvFireFighter(num_houses)

    o_n = env.get_obs()
    # observation is the state for each figherfighter ex: [[1,0],[1,1],[1,0]]



    # list of agent's reward
    agent_reward_v = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
    # output
    agent_reward_op = [tf.summary.scalar('agent' + str(i) + '_reward', agent_reward_v[i]) for i in range(3)]


    # list of agent's action
    agent_a1 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]

    agent_a1_op = [tf.summary.scalar('agent' + str(i) + '_action_1', agent_a1[i]) for i in range(3)]


    reward_100 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]

    reward_100_op = [tf.summary.scalar('agent' + str(i) + '_reward_l100_mean', reward_100[i]) for i in range(3)]



    reward_1000 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]

    reward_1000_op = [tf.summary.scalar('agent' + str(i) + '_reward_l1000_mean', reward_1000[i]) for i in range(3)]



    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    sess.run([agent1_actor_target_init, agent1_critic_target_init,

              agent2_actor_target_init, agent2_critic_target_init,

              agent3_actor_target_init, agent3_critic_target_init])





    agent1_memory = ReplayBuffer(100000)

    agent2_memory = ReplayBuffer(100000)

    agent3_memory = ReplayBuffer(100000)



    # e = 1



    reward_100_list = [[], [], []]

    # original is 1000000

    for i in range(1000):

        if i % 1000 == 0:

            o_n = env.reset() # return fire level


        agent1_action, agent2_action, agent3_action = get_agents_action(o_n, sess, noise_rate=0.2)



        #三个agent的行动
        agent1_action = np.squeeze(agent1_action, axis = 0)
        agent2_action = np.squeeze(agent2_action, axis = 0)
        agent3_action = np.squeeze(agent3_action)

        action1 = np.random.choice(np.arange(len(agent1_action)), p=agent1_action)
        action2 = np.random.choice(np.arange(len(agent2_action)), p=agent2_action)
        action3 = np.random.choice(np.arange(len(agent3_action)), p=agent3_action)

        a = [action1, action2, action3]
        print("action of agent is " + str(a))
        # global reward as the reward for each agent
        r_n = [env.step(a) for _ in range(3)]
        o_n_next = env.firelevel

        for agent_index in range(3):

            reward_100_list[agent_index].append(r_n[agent_index])

            reward_100_list[agent_index] = reward_100_list[agent_index][-1000:]



        agent1_memory.add(np.vstack([o_n[0], o_n[1], o_n[2]]),

                          np.vstack([agent1_action, agent2_action, agent3_action]),

                          r_n[0], np.vstack([o_n_next[0], o_n_next[1], o_n_next[2]]), False)



        agent2_memory.add(np.vstack([o_n[1], o_n[2], o_n[0]]),

                          np.vstack([agent2_action, agent3_action, agent1_action]),

                          r_n[1], np.vstack([o_n_next[1], o_n_next[2], o_n_next[0]]), False)



        agent3_memory.add(np.vstack([o_n[2], o_n[0], o_n[1]]),

                          np.vstack([agent3_action, agent1_action, agent2_action]),

                          r_n[2], np.vstack([o_n_next[2], o_n_next[0], o_n_next[1]]), False)


        # original is 50000
        if i > 10:

            # e *= 0.9999

            # agent1 train

            train_agent(agent1_ddpg, agent1_ddpg_target, agent1_memory, agent1_actor_target_update,

                        agent1_critic_target_update, sess, [agent2_ddpg_target, agent3_ddpg_target])



            train_agent(agent2_ddpg, agent2_ddpg_target, agent2_memory, agent2_actor_target_update,

                        agent2_critic_target_update, sess, [agent3_ddpg_target, agent1_ddpg_target])



            train_agent(agent3_ddpg, agent3_ddpg_target, agent3_memory, agent3_actor_target_update,

                        agent3_critic_target_update, sess, [agent1_ddpg_target, agent2_ddpg_target])

            print("trial " + str(i))
            print("fire level at trial " + str(i) + " is " + str(env.firelevel))
            print("reward " + str(env.get_obs()) + "\n\n")




        o_n = o_n_next


    sess.close()