"""
cartpole_v0 re-inforcement learning
"""
import math
from random import uniform
import numpy as np
import gym
from gym import wrappers

def get_observation_space_markers(observation_space, num_elements_per_dimension, scaling = 100):
    """
    given an observation_space, comes up with a dictionary for each dimension of the observation space
        each dictionary shows the size of each segment and whether we are using a logarithmic scale
    """

    os_high = observation_space.high
    os_low = observation_space.low
    num_dimensions = observation_space.shape[0]

    #os_segments = np.zeros(shape=(num_dimensions,num_elements_per_dimension))
    os_markers = [{} for _ in range(num_dimensions)]

    for i in range(num_dimensions):
        os_marker = os_markers[i]
        high = os_high[i]
        low = os_low[i]
        use_logarithmic_scale = False
        if high - low > 1e5:
            use_logarithmic_scale = True #we want to use log scale if the numbers are large
            high = math.copysign(math.log10(abs(high)), high)
            low = math.copysign(math.log10(abs(low)), low)

        shift = 0 # records how much to shift the value, to get zero based indices
        if low < 0:
            if use_logarithmic_scale:
                shift = abs(low ) / scaling
            else:
                shift = abs(low)
        os_marker['use_logarithmic_scale'] = use_logarithmic_scale
        if use_logarithmic_scale:
            segment_size = (high - low) / (num_elements_per_dimension * scaling)
        else:
            segment_size = (high - low) / (num_elements_per_dimension)
        os_marker['segment_size'] = segment_size
        os_marker['shift'] = shift
        os_marker['num_segments'] = num_elements_per_dimension

    return os_markers


def back_propagate(q_space, cells_visited, final_index, gamma, alpha_space, display_back_propagate):
    """
    given a qspace and list of cells visited (in order), back propagates the q values
    """

    print("back propagating: ", final_index)
    index_ = final_index
    alpha = 0.5
    for index, action, reward in reversed(cells_visited):
        if display_back_propagate:
            print("before: ", index, q_space[index], action)

        #alpha = 1 / alpha_space[index]
        sample = reward + gamma * max(q_space[index_])
        q_space[index][action] = (1 - alpha) * q_space[index][action] + alpha * sample
        #q_space[index][action] = sample # ignoring previous values
        #reward = backprop_decay * reward # why keep reward 0? since this is continuous space, i am back propagating the reward
        index_ = index
        if display_back_propagate:
            print("after: ", q_space[index])
        #break; # just do the last cell


def get_index(observation_space_markers, observation):
    """given an observation, returns a n-element vector that is the index into the q-value space"""
    num_dimensions = observation.shape[0]
    index = np.zeros(shape=(num_dimensions,), dtype=int)
    for i in range(num_dimensions):
        os_marker = observation_space_markers[i]
        obs_val = observation[i]
        segment_size = os_marker['segment_size']
        shift = os_marker['shift']
        index_i = math.floor( (obs_val + shift) / segment_size)
        if index_i >= os_marker['num_segments']:
            index[i] = os_marker['num_segments'] - 1
        elif index_i < 0:
            index[i] = 0
        else:
            index[i] = index_i

        #print("index = ", index_i, index[i])

    return tuple(index)


def main():
    """
        main entry point
        q learning
            sample = R(s, a, s_) + gamma * max(Q(s_, a_))
            Q(s, a) = (1 - alpha) * Q(s, a) + alpha * sample
    """
    env = gym.make('CartPole-v0')
    #env = wrappers.Monitor(env, './tmp/cartpole-experiment-1', )
    num_episodes = 100 # number of episodes we'll play
    max_steps_per_episode = 200
    num_elements_per_dimension = 7 # number of discrete pieces to divide each dimension of the observation space into
    done_reward = -200 # we want to give a large penalty if we exit the game
    display_back_propagate = False

    os_markers = get_observation_space_markers(env.observation_space, num_elements_per_dimension)
    q_space_shape = [os_marker['num_segments'] for os_marker in os_markers]
    alpha_space = np.zeros(shape=tuple(q_space_shape))

    q_space_shape.append(env.action_space.n) # the last dimension is the action space
    q_space = np.zeros(shape=tuple(q_space_shape))
    #q_space = np.load('./q_space.npy')


    gamma = 0.6 # discount factor. higher value means we still value old data
    backprop_decay = 0.8

    for i_episode in range(num_episodes):
        total_reward, reward = 0, 0
        observation = env.reset()
        index = get_index(os_markers, observation)

        cells_visited = []

        for t in range(max_steps_per_episode):
            env.render()

            alpha_space[index] += 1
            alpha = 1 / alpha_space[index]
            rand_number = uniform(0, 1)
            if rand_number < alpha:
                action = env.action_space.sample()
            else:
                argmax_a_ = np.argmax(q_space[index])
                action = argmax_a_

            observation_, reward, done, info = env.step(action)

            total_reward += reward

            index_ = get_index(os_markers, observation_)
            if index_ == index:
                #print("state did not change.")
                pass
            else:
                cells_visited.append((index, action, reward))
                # if not done, update q_values
                sample = reward + gamma * max(q_space[index_]) # changed reward to total_reward
                q_space[index][action] = (1 - alpha) * q_space[index][action] + alpha * sample
                #print(action, rand_number < alpha, reward, index, q_space[index])


            if done:
                print("Episode {} finished after {} timesteps. learning rate (alpha) = {}"
                    .format(i_episode, t+1, alpha))

                reward = total_reward + done_reward
                cells_visited[-1] = (index, action, reward)
                if i_episode == num_episodes - 1:
                    display_back_propagate = True
                    np.save("./q_space", q_space)
                if alpha > 0 and t < max_steps_per_episode - 1:
                    back_propagate(q_space, cells_visited, index_, gamma, alpha_space, display_back_propagate) # directly updates q_space
                break


            observation = observation_
            index = index_

if __name__ == '__main__':
    main()
