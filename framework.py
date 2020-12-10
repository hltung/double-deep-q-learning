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
    
    num_steps = dqn_rewards.size
    num_avging = 50
    
    avg_dqn_reward = np.zeros(num_steps - num_avging)
    #avg_ddqn_reward = np.zeros(num_steps - num_avging)
    
    for i in range(num_steps - num_avging):
        avg_dqn_reward[i] = np.mean(dqn_rewards[i:i+50])
        #avg_ddqn_reward[i] = np.mean(ddqn_reward[i:i+50])
    
    fig, ax = plt.subplots()
    x_values = list(range(1, num_steps - num_avging + 1))
    ax.plot(x_values, avg_dqn_reward, label='dqn rewards')
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
    num_steps = 5000
    rwd_list = []
        
    while len(rwd_list) < num_steps:
        state = env.reset()
        done = False
        print('new game, step:', len(rwd_list))
        
        while not done or len(rwd_list) < num_steps:
            # TODO:
            # 1) use model to generate probability distribution over next actions
            # 2) sample from this distribution to pick the next action
            action = 0
            state = tf.reshape(state, [tf.size(state)])
            if np.random.uniform() < model.epsilon:
                # action = tf.convert_to_tensor(np.random.randint(0, model.num_actions))
                action = env.action_space.sample()          
            else:
                # state appears to be a tensor of dimensions [250, 160, 3]
                # reshaping for now
                # tf.reshape(state, [tf.size(state)]
                q = model.call(tf.expand_dims(state, axis=0))
                action = tf.math.argmax(tf.squeeze(q)).numpy()
            prev_state = state
            state, rwd, done, _ = env.step(action)
            rwd_list.append(rwd)
            state = tf.reshape(state, [tf.size(state)])
            model.buffer.push(prev_state, action, state, rwd)
            train(env, model)
            model.epsilon = model.epsilon * model.epsilon_update
        env.reset()
    return rwd_list


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
        gradients = tape.gradient(loss_val, model.trainable_variables)
        # no idea if this works
        if isinstance(model, DDQN):
            model.Q_target = tf.keras.model.clone(model.Q)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

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

    num_games = 650
    for i in range(num_games):
        dqn_rwd = generate_trajectory(env, dqn_model)
        dqn_rwds.append(dqn_rwd)
        ddqn_rwd = generate_trajectory(env, ddqn_model)
        ddqn_rwds.append(ddqn_rwd)

    env.close()
    
    print("DQN rewards")
    print(dqn_rwds)
    print("DDQN Rewards")
    #print(ddqn_rwds)

    # TODO: Visualize your rewards.
    visualize_data(np.array(dqn_rwds), np.array(ddqn_rwds))


if __name__ == '__main__':
    main()
