import os
import gym
import numpy as np
import tensorflow as tf
from replay_memory import ReplayMemory

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class DQN(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The Reinforce class that inherits from tf.keras.Model
        The forward pass calculates the policy for the agent given a batch of states.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this,
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.batch_size = 128
        self.epsilon = 0.7
        self.epsilon_update = 0.995
        

        # TODO: Define network parameters and optimizer
        
        self.buffer = ReplayMemory(100000)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, 10000, 0.1)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        hidden_sz1 = 256 
        hidden_sz2 = 128
        
        self.Q_1 = tf.keras.layers.Dense(hidden_sz1)
        self.Q_2 = tf.keras.layers.Dense(hidden_sz2)
        self.Q_3 = tf.keras.layers.Dense(self.num_actions)


    def call(self, states):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a Q-value tensor of shape [batch_size, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [batch_size, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [batch_size, num_actions] matrix representing the Q-values over actions
        for each state in the episode
        """
        # TODO: implement this ~
        l1 = tf.nn.relu(self.Q_1(states))
        l2 = tf.nn.relu(self.Q_2(l1))
        qVals = self.Q_3(l2)
        return qVals
        # return tf.argmax(qVals, 1)

    def loss(self, states, actions, next_states, rewards, discount_rate=.99):
        """
        Computes the loss for the agent.

        :param states: A batch of states of shape [batch_size, state_size]
        :param actions: History of actions taken at each timestep of the episode (represented as an [batch_size] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [batch_size] array)
        :return: loss, a Tensorflow scalar
        """
        # TODO: implement this
        actions = tf.cast(actions, tf.int64)
        a = tf.stack([tf.range(states.shape[0],dtype=tf.int64), actions], axis=1)
        qVals = tf.gather_nd(self.call(states), a) # [batch_size] q-values for each action
        nextVals = tf.reduce_max(self.call(next_states), axis=1) # max of q-values [batch_size, num_actions] across num_actions
        targetVals = rewards + (discount_rate*nextVals)
        loss = tf.reduce_sum(tf.math.square(qVals - targetVals))
        return loss
