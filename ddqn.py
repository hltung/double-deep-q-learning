import os
import gym
import numpy as np
import tensorflow as tf
from replay_memory import ReplayMemory


# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class DDQN(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The Reinforce class that inherits from tf.keras.Model
        The forward pass calculates the policy for the agent given a batch of states.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this,
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(DDQN, self).__init__()
        self.num_actions = num_actions
        self.batch_size = 64
        self.epsilon = 0.7
        self.epsilon_update = 0.9

        # TODO: Define network parameters and optimizer
        self.buffer = ReplayMemory(1000)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, 500, 0.1)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        self.Q = tf.keras.layers.Dense(self.num_actions)
        self.Q_target = tf.keras.layers.Dense(self.num_actions)
    
    def call(self, states):
        qVals = self.Q(states)
        return qVals

    def call_target(self, states, next_states, rewards, discount_rate=.9):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        for each state in the episode
        """
        # TODO: implement this ~
        fut_actions = tf.argmax(self.Q(next_states))
        fut_actions_ind = tf.stack([tf.range(next_states.shape[0]), fut_actions], axis=1)
        q_next = tf.gather_nd(self.Q_target(next_states), fut_actions)
        qVals = rewards + discount_rate * q_next
        return qVals


    def loss(self, states, actions, next_states, rewards, discount_rate=.9):
        """
        Computes the loss for the agent. Make sure to understand the handout clearly when implementing this.

        :param states: A batch of states of shape [episode_length, state_size]
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a Tensorflow scalar
        """
        # TODO: implement this
        # Hint: Use gather_nd to get the probability of each action that was actually taken in the episode.
        a = tf.stack([tf.range(states.shape[0]), actions], axis=1)
        tf.stop_gradient(self.Q_target)
        qVals = tf.gather_nd(self.call_target(states, next_states, rewards), a)
        return tf.reduce_sum(tf.math.square(qVals - self.call(states)))
