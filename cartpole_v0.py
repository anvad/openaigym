"""
cartpole_v0 re-inforcement learning

Episode 123 finished after 200 timesteps. learning rate (alpha) = 0.0004084967320261438
reward at done:  1.0
running average of reward per episode:  196.3

n = 3
Episode 61 finished after 174 timesteps. learning rate (alpha) = 0.1111111111111111
reward at done:  1.0
running average of reward per episode:  195.7

I'm going to assume that you have an (ndims,) vector of indices specifying some point p, and you want an (m, ndims) array of indices corresponding to the locations of every adjacent element in the array (including diagonally adjacent elements).
Starting out with your indexing vector p, you want to offset each element by every possible combination of -1, 0 and +1. This can be done by using np.indices to generate an (m, ndims) array of offsets, then adding these offsets to p.
You might want to exclude point p itself (i.e. where offset == np.array([0, 0, ..., 0]), and you may also need to exclude out-of-bounds indices.
hvwwtwzdtp255jehzgu6tihbujnyv6yn6rswvpm4gzo3jawqhewq
"""

import math
from random import uniform
import collections
import sys
import ast
import numpy as np
import gym
from gym import wrappers


class observation_space:
    def __init__(self, high, low):
        self.high = high
        self.low = low

def get_observation_space_markers(observation_space, num_elements_per_dimension, scaling=1):
    """
    given an observation_space, comes up with a list of dictionaries
    each dictionary represents one dimension of the observation space
    each dictionary shows the size of each segment and whether we are using a sigmoid scale
    """

    os_high = observation_space.high
    os_low = observation_space.low
    num_dimensions = observation_space.high.shape[0]

    os_markers = [{} for _ in range(num_dimensions)]

    for i in range(num_dimensions):
        os_marker = os_markers[i]
        high = os_high[i]
        low = os_low[i]
        use_sigmoid_scale = False
        if high - low > 1e30:
            use_sigmoid_scale = True #we want to use sigmoid scale if the numbers are large
            high = 1
            low = 0

        shift = 0 # records how much to shift the value, to get zero based indices
        if low < 0:
            shift = abs(low)
        os_marker['use_sigmoid_scale'] = use_sigmoid_scale
        segment_size = (high - low) / (num_elements_per_dimension)
        os_marker['segment_size'] = segment_size
        os_marker['shift'] = shift
        os_marker['num_segments'] = num_elements_per_dimension

    return os_markers


def get_index(observation_space_markers, observation):
    """given an observation, returns a n-element vector that is the index into the q-value space"""
    num_dimensions = observation.shape[0]
    index = np.zeros(shape=(num_dimensions,), dtype=int)
    for i in range(num_dimensions):
        os_marker = observation_space_markers[i]
        obs_val = observation[i]
        if os_marker['use_sigmoid_scale']:
            obs_val = 1 / (1 + math.exp(-obs_val))
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


def main(args):
    """
        main entry point
        q learning
            sample = R(s, a, s_) + gamma * max(Q(s_, a_))
            Q(s, a) = (1 - alpha) * Q(s, a) + alpha * sample
    """
    if len(args) == 0:
        q_learning()
    elif len(args) == 2:
        apply_learned_q_values(args[0], args[1])
    else:
        apply_learned_q_values_2(4)

def apply_learned_q_values_2(num_elements_per_dimension):
    q_space_str = '[[[[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]], [[[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [2.3636896239274665, 0.0], [-9.246607699955423, -0.49098630063633175], [-34.801766815449334, 2.466097125649913]], [[0.0, 0.0], [0.0, 0.0], [-3.6200593008410884, 2.314332088353076], [-172.25784566066702, 2.4593515335735288]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[-13.983768086297104, -39.603960396039604], [-60.386119719115335, -0.031401772050961285], [2.484897578429833, -11.14401818905544], [-55.693349241913566, 2.4862987342296883]], [[0.0, 0.0], [2.4872828937281533, -0.23645000745227376], [-36.001966571615526, 2.2708256696463955], [-145.64151957255763, 2.4894592644987643]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[2.4917444721457653, -115.61027114631268], [2.489201139982746, -67.65208302667571], [2.4970445750432178, -1.9692154994292173], [0.0, 0.0]], [[2.493084716294367, -31.210501653951827], [-4.645354177476377, 2.4973905901555615], [-49.34672738620378, 2.4363356351730903], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[-19.23610137594166, 1.422955040898907], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[1.8986286366903287, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]], [[[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.24832848239462557], [2.265701840120602, -0.9135147055278313]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [-130.99748734282602, 1.1509289939244858]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [2.489767798031736, -116.47417084136045], [2.4280647277405842, -7.365215021153919], [-2.8149473902096567, 2.4672276070475117]], [[0.0, 0.0], [-6.834931857737478, 2.4902556113030987], [-41.48101909062802, 2.472459384757004], [-48.52382045202813, 2.4541520838413273]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[2.4769664149754225, -92.65494604622933], [2.4863357549744194, -49.93633733143474], [-2.2299145017776314, 2.4709667143277994], [0.0, 0.0]], [[2.4901405772785727, -1.4680866222813884], [-52.968445017382855, 2.4885175767619856], [-119.0826498646617, 0.5622127635091512], [0.0, -98.9925169989638]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[2.4841991559737737, -186.12871287128715], [2.1291324032604098, 1.413618596315585], [0.0, 0.0], [0.0, 0.0]], [[2.3035026262273646, -13.540767032295046], [1.9227441115572859, -10.550598116906428], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]], [[[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]]]'
    q_space = np.array(ast.literal_eval(q_space_str))
    apply_learned_q_values_inner(q_space, num_elements_per_dimension)

def apply_learned_q_values_inner(q_space, num_elements_per_dimension):
    env = gym.make('CartPole-v0')
    num_episodes = 10000 # number of episodes we'll play
    max_steps_per_episode = 200
    num_last_few_episodes = 20

    last_few_rewards = collections.deque(maxlen=num_last_few_episodes)

    # get initial OS markers
    os_markers = get_observation_space_markers(env.observation_space, num_elements_per_dimension)



    for i_episode in range(num_episodes):
        total_reward, reward = 0, 0
        observation = env.reset()
        index = get_index(os_markers, observation)

        for t in range(max_steps_per_episode):
            argmax_a_ = np.argmax(q_space[index])
            action = argmax_a_

            observation_, reward, done, info = env.step(action)

            total_reward += reward

            index_ = get_index(os_markers, observation_)

            if done:
                print("\nEpisode {} finished after {} timesteps."
                    .format(i_episode, t+1))
                break

            observation = observation_
            index = index_

        #after an episode is over
        # update running average
        last_few_rewards.append(total_reward)
        running_average = sum(last_few_rewards)/num_last_few_episodes
        print("running average of reward per episode: ", running_average)
        if running_average >= 195.0:
            np.save('./q_space.npy', q_space)
            break


def apply_learned_q_values(q_space_file_name, num_elements_per_dimension):
    """
    given a numpy array file, uses the learned q-values to apply the best policy
    """

    num_elements_per_dimension = int(num_elements_per_dimension)
    q_space = np.load(q_space_file_name) # './q_space.npy'
    print("q_space_file_name={}, num_elements_per_dimension={}".format(q_space_file_name, num_elements_per_dimension))

    apply_learned_q_values_inner(q_space, num_elements_per_dimension)



def q_learning():
    """
        q learning
            sample = R(s, a, s_) + gamma * max(Q(s_, a_))
            Q(s, a) = (1 - alpha) * Q(s, a) + alpha * sample
    """
    # all magic numbers and env specific details
    env = gym.make('CartPole-v0')
    #env = wrappers.Monitor(env, './tmp/cartpole-experiment-1', )
    num_episodes = 10000 # number of episodes we'll play
    max_steps_per_episode = 200
    num_elements_per_dimension = 4 # number of discrete pieces to divide each dimension of the observation space into
    done_reward = -200
    gamma = 0.6  # discount factor. higher value means we still value old data
    num_last_few_episodes = 20

    last_few_rewards = collections.deque(maxlen=num_last_few_episodes)

    # get initial OS markers
    os_markers = get_observation_space_markers(env.observation_space, num_elements_per_dimension)

    # set up q(s,a) and alpha spaces in numpy
    q_space_shape = [os_marker['num_segments'] for os_marker in os_markers]
    alpha_space = np.zeros(shape=tuple(q_space_shape)) # store alpha (learning rate) per state

    q_space_shape.append(env.action_space.n) # the last dimension is the action space
    q_space = np.zeros(shape=tuple(q_space_shape))
    #q_space = np.load('./q_space.npy')

    for i_episode in range(num_episodes):
        total_reward, reward = 0, 0
        observation = env.reset()
        index = get_index(os_markers, observation)

        cells_visited = [] # store chronological list of states visited

        num_same_state = 0 # init to 0, number of times we've remained in same state after taking action
        for t in range(max_steps_per_episode):
            #env.render()

            alpha = 100 / (100 + alpha_space[index])
            rand_number = uniform(0, 1)

            epsilon = 1 / (1 + alpha_space[index])
            if rand_number < epsilon: # note: here, alpha is doing double duty as a random action selector
                # perhaps we should replace this with an exploration function? or a neighborhood analysis?
                action = env.action_space.sample()
            else:
                argmax_a_ = np.argmax(q_space[index])
                action = argmax_a_

            observation_, reward, done, info = env.step(action)

            total_reward += reward

            index_ = get_index(os_markers, observation_)

            # update alpha and q values
            alpha_space[index] += 1

            if index_ == index:
                #print("state did not change.")
                print('.', end=' ')
                num_same_state += 1
            else:
                print('+', end=' ')
                cells_visited.append((index, action, reward))
                num_same_state = 0 # reset to 0
                #print(action, rand_number < alpha, reward, index, q_space[index])

            if done:
                print("\nEpisode {} finished after {} timesteps. learning rate (alpha) = {}"
                    .format(i_episode, t+1, alpha))
                print("reward at done: ", reward)
                reward = total_reward + done_reward # put a penalty for dying.
                # update q values of final state with new reward
                sample = reward # changed reward to total_reward
                q_space[index][action] = (1 - alpha) * q_space[index][action] + alpha * sample
                break
            else:
                sample = (reward) + gamma * max(q_space[index_]) # changed reward to total_reward
                q_space[index][action] = (1 - alpha) * q_space[index][action] + alpha * sample

            observation = observation_
            index = index_

        #after an episode is over
        # update running average
        last_few_rewards.append(total_reward)
        running_average = sum(last_few_rewards)/num_last_few_episodes
        print("running average of reward per episode: ", running_average)
        if running_average >= 195.0:
            np.save('./q_space.npy', q_space)
            break


if __name__ == '__main__':
    main(sys.argv[1:])
