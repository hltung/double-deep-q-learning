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
        self.batch_size = 128
        self.epsilon = 0.7
        self.min_epsilon = 0.05
        self.epsilon_update = 0.995
        
        # TODO: Define network parameters and optimizer
        self.buffer = ReplayMemory(10000)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, 9000, 0.1)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        hidden_sz1 = 256 
        hidden_sz2 = 128
        
        self.Q_1 = tf.keras.layers.Dense(hidden_sz1)
        self.Q_2 = tf.keras.layers.Dense(hidden_sz2)
        self.Q_3 = tf.keras.layers.Dense(self.num_actions)
        
        self.Q1_target = tf.keras.layers.Dense(hidden_sz1)
        self.Q1_target.trainable = False
        self.Q2_target = tf.keras.layers.Dense(hidden_sz2)
        self.Q2_target.trainable = False
        self.Q3_target = tf.keras.layers.Dense(self.num_actions)
        self.Q3_target.trainable = False
    
    def call(self, states):
        l1 = tf.nn.relu(self.Q_1(states))
        l2 = tf.nn.relu(self.Q_2(l1))
        qVals = self.Q_3(l2)
        return qVals
    
    def call_target(self, states):
        l1 = tf.nn.relu(self.Q1_target(states))
        l2 = tf.nn.relu(self.Q2_target(l1))
        qVals = self.Q3_target(l2)
        return qVals

    def get_qVals(self, states, next_states, rewards, discount_rate=.9):
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
        
        fut_actions = tf.dtypes.cast(tf.argmax(self.call(next_states), 1), tf.int64)
        fut_actions_ind = tf.stack([tf.range(next_states.shape[0],dtype=tf.int64), fut_actions], axis=1)
        q_target = tf.stop_gradient(self.call_target(next_states))
        q_next = tf.gather_nd(q_target, fut_actions_ind)
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
        actions = tf.cast(actions, tf.int64)
        a = tf.stack([tf.range(states.shape[0],dtype=tf.int64), actions], axis=1)
        qVals = self.get_qVals(states, next_states, rewards)
        policyQ = tf.gather_nd(self.call(states), a)
        return tf.reduce_sum(tf.math.square(qVals - policyQ))
        