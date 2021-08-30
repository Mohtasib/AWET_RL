import os
import pickle
import numpy as np
from random import sample
from tslearn.metrics import dtw

def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# This method is to sample num_demos from the demos dataset
def sample_n_demos(expert_demos, num_demos):
    if num_demos < len(expert_demos):
        demos = sample(range(len(expert_demos)),num_demos)
        new_demos = dict()
        for i in range(len(demos)):
            new_demos[f'demo{i}'] = expert_demos[f'demo{demos[i]}']
        return new_demos   
    else:
        return expert_demos

def calculate_q_values(data, gamma=0.99):
    """
    This method recieves the data in the form:
    data = { 'demo0':{'obs':demo0_obs, 'next_obs':demo0_next_obs, 'actions':demo0_actions, 'rewards':demo0_rewards, 'dones':demo0_dones},
             'demo1':{'obs':demo1_obs, 'next_obs':demo1_next_obs, 'actions':demo1_actions, 'rewards':demo1_rewards, 'dones':demo1_dones},
             . . .
             'demon':{'obs':demon_obs, 'next_obs':demon_next_obs, 'actions':demon_actions, 'rewards':demon_rewards, 'dones':demon_dones}}
    And returns the data in the form:
    data = { 'demo0':{'obs':demo0_obs, 'next_obs':demo0_next_obs, 'actions':demo0_actions, 'rewards':demo0_rewards, 'dones':demo0_dones, 'q_values':demo0_q_values},
             'demo1':{'obs':demo1_obs, 'next_obs':demo1_next_obs, 'actions':demo1_actions, 'rewards':demo1_rewards, 'dones':demo1_dones, 'q_values':demo1_q_values},
             . . .
             'demon':{'obs':demon_obs, 'next_obs':demon_next_obs, 'actions':demon_actions, 'rewards':demon_rewards, 'dones':demon_dones, 'q_values':demon_q_values}}
    """
    new_data = dict()
    for i in range(len(data)):
        demo = data[f'demo{i}']
        obs = demo['obs']
        next_obs = demo['next_obs']
        actions = demo['actions']
        rewards = demo['rewards']
        dones = demo['dones']
        
        # print(f'trajectory obs = {obs[10:15]}')
        # Check the data sizes:
        assert obs.shape[0] == actions.shape[0] == rewards.shape[0], "Error!, Incompatible sizes"

        q_values = np.zeros_like(rewards)
        for j in range(1, rewards.shape[0]+1):
            q_values[-j] = rewards[-j] + (1- dones[-j]) * gamma * q_values[-j+1]
        
        new_data[f'demo{i}'] = {'obs':obs, 'next_obs':next_obs, 'actions':actions, 'rewards':rewards, 'dones':dones, 'q_values':q_values}

    return new_data

def create_transitions(data):
    """
    This method recieves the data in the form:
    data = { 'demo0':{'obs':demo0_obs, 'next_obs':demo0_next_obs, 'actions':demo0_actions, 'rewards':demo0_rewards, 'dones':demo0_dones, 'q_values':demo0_q_values},
             'demo1':{'obs':demo1_obs, 'next_obs':demo1_next_obs, 'actions':demo1_actions, 'rewards':demo1_rewards, 'dones':demo1_dones, 'q_values':demo1_q_values},
             . . .
             'demon':{'obs':demon_obs, 'next_obs':demon_next_obs, 'actions':demon_actions, 'rewards':demon_rewards, 'dones':demon_dones, 'q_values':demon_q_values}}
    And returns the transitions in the form:
    transitions = { 'transition0':{'obs':transition0_obs, 'next_obs':transition0_next_obs, 'actions':transition0_actions, 'rewards':transition0_rewards, 'dones':transition0_dones, 'q_values':transition0_q_values},
                    'transition1':{'obs':transition1_obs, 'next_obs':transition1_next_obs, 'actions':transition1_actions, 'rewards':transition1_rewards, 'dones':transition1_dones, 'q_values':transition1_q_values},
                    . . .
                    'transitionm':{'obs':transitionm_obs, 'next_obs':transitionm_next_obs, 'actions':transitionm_actions, 'rewards':transitionm_rewards, 'dones':transitionm_dones, 'q_values':transitionm_q_values}}
    """
    transitions = dict()
    count = 0
    for i in range(len(data)):
        demo = data[f'demo{i}']
        obs = demo['obs']
        next_obs = demo['next_obs']
        actions = demo['actions']
        rewards = demo['rewards']
        dones = demo['dones']
        q_values = demo['q_values']

        for j in range(obs.shape[0]):
            transitions[f'transition{count}'] = {'obs':obs[j], 'next_obs':obs[j], 'actions':actions[j], 'rewards':rewards[j], 'dones':dones[j], 'q_values':q_values[j]}
            count += 1

    return transitions

def create_trajectories(data):
    trajectories = []
    for i in range(len(data)):
        demo = data[f'demo{i}']
        trajectory = demo['obs']
        trajectory = np.expand_dims(trajectory, axis=0)
        trajectories.append(trajectory)

    trajectories = np.vstack(trajectories)

    return trajectories

def calculate_trajectory_similarity(trajectory, trajectories, sim_test_point):
    sim = []
    for i in range(trajectories.shape[0]):
        sim.append(dtw(trajectory, trajectories[i, :sim_test_point, :]))

    # return np.average(np.array(sim))
    return min(np.array(sim))

def calculate_similarity_threshold(trajectories, sim_test_point):
    sim = []
    M = trajectories.shape[0] # Number of trajectories
    for i in range(M-1):
        for j in range(i+1):
            sim.append(dtw(trajectories[i, :sim_test_point, :], trajectories[j, :sim_test_point, :]))
    sim = np.array(sim)
    sim_thr = 2 * np.sum(sim) / ((M*M) - M)
    return sim_thr

# data = calculate_q_values(load_data('Demos_data/Pendulum_Demos.pkl'))
# transitions = create_transitions(data)

# print(len(data))
# print(len(transitions))


# trajectories = create_trajectories(load_data('/home/abdalkarim/MyProjects/test_myrl/Demos_data/Pendulum_Dense_Demos_100.pkl'))
# sim = calculate_trajectory_similarity(trajectories[1][:25], trajectories[:,:25,:])
# print(trajectories.shape)
# print(sim)