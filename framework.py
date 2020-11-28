import os
import sys
import gym
from pylab import *
import numpy as np
import tensorflow as tf
from dqn import DQN
from ddqn import DDQN


def visualize_data(total_rewards):
    """
    Takes in array of rewards from each episode, visualizes reward over episodes

    :param total_rewards: List of rewards from all episodes
    """

    x_values = arange(0, len(total_rewards), 1)
    y_values = total_rewards
    plot(x_values, y_values)
    xlabel('episodes')
    ylabel('cumulative rewards')
    title('Reward by Episode')
    grid(True)
    show()


def discount(rewards, discount_factor=.99):
    """
    Takes in a list of rewards for each timestep in an episode, and
    returns a list of the discounted rewards for each timestep.
    Refer to the slides to see how this is done.

    :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
    :param discount_factor: Gamma discounting factor to use, defaults to .99
    :returns: discounted_rewards: list containing the discounted rewards for each timestep in the original rewards list
    """
    # TODO: Compute discounted rewards
    
    discounted = np.zeros(len(rewards))
    discounted[-1] = rewards[len(rewards) - 1] 
    for i in range(len(rewards) - 2, -1, -1):
        discounted[i] = discounted[i+1] * discount_factor + rewards[i]
    return tf.convert_to_tensor(discounted, dtype=float32)

def generate_trajectory(env, model):
    """
    Generates lists of states, actions, and rewards for one complete episode.

    :param env: The openai gym environment
    :param model: The model used to generate the actions
    :returns: A tuple of lists (states, actions, rewards), where each list has length equal to the number of timesteps in the episode
    """
    state = env.reset()
    done = False
    
    while not done:
        # TODO:
        # 1) use model to generate probability distribution over next actions
        # 2) sample from this distribution to pick the next action
        dist = (model.call(tf.expand_dims(state, axis=0))[0]).numpy()
        dist = dist / np.sum(dist)
        action = np.random.choice(model.num_actions, p=dist)
        prev_state = state
        state, rwd, done, _ = env.step(action)
        model.buffer.push(prev_state, action, state, rwd)
        train(env, model)


def train(env, model):
    """
    This function should train your model for one episode.
    Each call to this function should generate a complete trajectory for one
    episode (lists of states, action_probs, and rewards seen/taken in the episode), and
    then train on that data to minimize your model loss.
    Make sure to return the total reward for the episode

    :param env: The openai gym environment
    :param model: The model
    :returns: The total reward for the episode
    """
    training_threshold = 1.5 * model.batch_size

    # TODO:
    # 1) Use generate trajectory to run an episode and get states, actions, and rewards.
    # 2) Compute discounted rewards.
    # 3) Compute the loss from the model and run backpropagation on the model.
    if len(model.buffer) > training_threshold:
        with tf.GradientTape() as tape:
            experiences = model.buffer.sample(model.batch_size)

            #check this chunk of code works
            states = tf.concat([exps[0] for exps in experiences], 0)
            actions = tf.concat([exps[1] for exps in experiences], 0)
            next_states = tf.concat([exps[2] for exps in experiences], 0)
            rewards = tf.concat([tf.convert_to_tensor(exps[3]) for exps in experiences], 0)

            loss_val = model.loss(states, actions, next_states, rewards)
        gradients = tape.gradient(loss_val, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(env, model):
    """
    Test how the model does with 1 game or something
    
    :param env: The openai gym environment
    :param model: The model
    """

def main():
    env = gym.make("VideoPinball-v0")
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Initialize model
    dqn_model = DQN(state_size, num_actions) 
    ddqn_model = DDQN(state_size, num_actions)

    # TODO: 
    # 1) Train your model for 650 episodes, passing in the environment and the agent. 
    # 2) Append the total reward of the episode into a list keeping track of all of the rewards. 
    # 3) After training, print the average of the last 50 rewards you've collected.
    num_games = 650
    for i in range(num_games):
        generate_trajectory(env, ddqn_model)
    reward_avg = np.sum(reward_list[-50:]) / 50
    print(reward_avg)
    env.close()

    # TODO: Visualize your rewards.
    visualize_data(reward_list)


if __name__ == '__main__':
    main()
