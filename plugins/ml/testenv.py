# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 13:02:41 2018

@author: Bart
"""
import numpy as np
from critic import CriticNetwork
from actor import ActorNetwork
from ReplayMemory import ReplayMemory
import tensorflow as tf
from OU import OrnsteinUhlenbeckActionNoise


class Env():
    def __init__(self, players):
        self.players = players
        self.observation_space = 1
        self.action_space = 1
        self.reset()

    def reset(self):
        self.observation = np.random.normal(scale=6, size=(self.players,1))
        low = np.where(self.observation<-10)[0]
        high = np.where(self.observation>10)[0]
        self.observation[low] = -10.
        self.observation[high] = 10.
        return self.observation

    def step(self, actions):
        answer = np.sum(self.observation) * np.ones(self.observation.shape)
        reward = np.sum(-1 * np.abs(actions-answer)) /5 * np.ones((self.players,1))
#        print(np.sum(self.observation), actions)

        return answer, reward, True


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    # episode_ave_max_q = tf.Variable(0.)
    # tf.summary.scalar("Qmax Value", episode_ave_max_q)
    summary_vars = [episode_reward]# , episode_ave_max_q]
    summary_ops = tf.summary.merge_all()
    return summary_ops, summary_vars


def train(sess, env, actor, critic, actor_noise):
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./testsummaries/', sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()
    # Initialize replay memory
    replay_buffer = ReplayMemory(10000)

    episodes = 200000
    for i in range(episodes):
        s = env.reset()

        ep_reward = 0
        for j in range(1):

            # Added exploration noise
            #a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
#            print('state ', s)
#            print('prediction ', actor.predict(s.reshape((1,5,1))))
            a = actor.predict(s.reshape((1,5,1))) #+ actor_noise()
#            print('action ', a)
#            print('noise', actor_noise())
            s2, r, terminal= env.step(a)

            replay_buffer.add(s, a.reshape((env.players,1)), r, s2, terminal)

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
#            print('buffer_size', replay_buffer.size())
            if replay_buffer.count() > 32:
                batch = replay_buffer.getBatch(32)


                states = np.asarray([seq[0] for seq in batch])
                actions = np.asarray([seq[1] for seq in batch])
                rewards = np.asarray([seq[2] for seq in batch])
                new_states = np.asarray([seq[3] for seq in batch])
                dones = np.asarray([seq[4] for seq in batch])
                y_t = rewards.copy()

                target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

                #Compute the target values
                for k in range(len(batch)):
                    if dones[k]:
                        y_t[k] = rewards[k]
                    else:
                        gamma = 0.98
                        # print(rewards[k].shape, target_q_values[k].shape, y_t[k].shape)
                        y_t[k] = rewards[k] + gamma * target_q_values[k]

                if (1):
                    # self.loss += self.critic.model.train_on_batch([states, actions], y_t)
                    actions_for_grad = actor.model.predict(states)
                    grads = critic.gradients(states, actions_for_grad)
#                    print('shapes' ,states.shape, actions.shape, y_t.shape )
                    critic.train(states, actions, y_t)
                    actor.train(states, grads)
                    actor.update_target_network()
                    critic.update_target_network()


            ep_reward += r

#            if terminal:
#                summary_str = sess.run(summary_ops, feed_dict={
#                    summary_vars[0]: np.sum(ep_reward),
#                })
#                writer.add_summary(summary_str, i)
#                writer.flush()
#                break
            if i%100==0:
                print("episode {} reward: {}".format(i, np.sum(ep_reward)))
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: np.sum(ep_reward),
                })
                writer.add_summary(summary_str, i)
                writer.flush()
                break



def main():
    tf.reset_default_graph()
    with tf.Session() as sess:
        env = Env(players=5)
        np.random.seed(1)
        tf.set_random_seed(1)
        state_dim = env.observation_space
        action_dim = env.action_space

        # Ensure action bound is symmetric
        crlr = 0.001
        aclr = 0.001
        tau = 0.001
        actor = ActorNetwork(sess, state_dim, action_dim, 32, tau, aclr)
        critic = CriticNetwork(sess, state_dim, action_dim, 32, tau, crlr)
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros((env.players, action_dim)))

        train(sess, env, actor, critic, actor_noise)


if __name__ == '__main__':
    main()


