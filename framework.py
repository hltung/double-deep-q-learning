import os
import sys
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dqn import DQN
from ddqn import DDQN


def visualize_data(dqn_rewards, ddqn_rewards):
    """
    Takes in array of rewards from each episode, visualizes reward over episodes

    :param total_rewards: List of rewards from all episodes
    """
        
    fig, ax = plt.subplots()
    x_values = list(range(1, dqn_rewards.size + 1))
    ax.plot(x_values, dqn_rewards, label='dqn rewards')
    #ax.plot(x_values, avg_ddqn_reward, label='ddqn rewards')
    plt.xlabel('episodes')
    plt.title('Reward by Episode')
    plt.legend()
    plt.show()



def generate_trajectory(env, model):
    """
    Generates lists of states, actions, and rewards for one complete episode.

    :param env: The openai gym environment
    :param model: The model used to generate the actions
    :returns: A tuple of lists (states, actions, rewards), where each list has length equal to the number of timesteps in the episode
    """

    state = env.reset()
    done = False
    cumulative_rwd = 0
    
    while not done:
        # TODO:
        # 1) use model to generate probability distribution over next actions
        # 2) sample from this distribution to pick the next action
        action = 0
        if np.random.uniform() < model.epsilon:
            action = env.action_space.sample()         
        else:
            q = model.call(tf.expand_dims(state, axis=0))
            action = tf.math.argmax(tf.squeeze(q))
        prev_state = state
        state, rwd, done, _ = env.step(action)
        cumulative_rwd = cumulative_rwd + rwd
        model.buffer.push(prev_state, action, state, rwd)
        train(env, model)
        model.epsilon = model.epsilon * model.epsilon_update
    return cumulative_rwd


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
    training_threshold = model.batch_size
    tau = 0.9

    # TODO:
    # 1) Use generate trajectory to run an episode and get states, actions, and rewards.
    # 2) Compute discounted rewards.
    # 3) Compute the loss from the model and run backpropagation on the model.
    if len(model.buffer) > training_threshold:
        with tf.GradientTape() as tape:
            experiences = model.buffer.sample(model.batch_size)

            #check this chunk of code works may need to edit lol
            states = tf.stack([exps[0] for exps in experiences], 0)
            actions = tf.stack([exps[1] for exps in experiences], 0)
            next_states = tf.stack([exps[2] for exps in experiences], 0)
            rewards = tf.stack([exps[3] for exps in experiences], 0)

            loss_val = model.loss(states, actions, next_states, rewards)
        
        # no idea if this works
        gradients = tape.gradient(loss_val, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if isinstance(model, DDQN):    
            # model.optimizer.apply_gradients(tape.gradient(loss_val,model.Q_1),model.Q_1)
            # model.optimizer.apply_gradients(tape.gradient(loss_val,model.Q_2),model.Q_2)
            # model.optimizer.apply_gradients(tape.gradient(loss_val,model.Q_3),model.Q_3)
            model.Q1_target.set_weights(model.Q_1.get_weights())
            model.Q2_target.set_weights(model.Q_2.get_weights())
            model.Q3_target.set_weights(model.Q_3.get_weights())

def main():
    env = gym.make("SpaceInvaders-ram-v0")
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Initialize model
    dqn_model = DQN(state_size, num_actions) 
    ddqn_model = DDQN(state_size, num_actions)

    # TODO: 
    # 1) Train your model for 650 episodes, passing in the environment and the agent. 
    # 2) Append the total reward of the episode into a list keeping track of all of the rewards. 
    # 3) After training, print the average of the last 50 rewards you've collected.
    
    dqn_rwds = []
    ddqn_rwds = []
    
    print('start train')

    num_games = 600
    for i in range(num_games):
        if i%10 == 0:
            print('step:', i)
            print(dqn_model.epsilon)
        ddqn_rwd = generate_trajectory(env, ddqn_model)
        ddqn_rwds.append(ddqn_rwd)
        dqn_rwd = generate_trajectory(env, dqn_model)
        dqn_rwds.append(dqn_rwd)
        
    env.close()
    
    print("DQN rewards")
    print(dqn_rwds)
    print("DDQN Rewards")
    print(ddqn_rwds)

    # TODO: Visualize your rewards.
    visualize_data(np.array(dqn_rwds), np.array(ddqn_rwds))


if __name__ == '__main__':
    main()
