import os
import pickle
import numpy as np
from random import sample
from tslearn.metrics import dtw

from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

def listdirs(path):
    dirlist = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    dirlist.sort()
    return dirlist
    
def save_pickle_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pickle_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# This method is to sample num_demos from the demos dataset
def sample_n_demos(expert_demos, num_demos):
    if num_demos < len(expert_demos):
        demos = sample(range(len(expert_demos)),num_demos)
        new_demos = dict()
        demo_names = list(expert_demos.keys())
        for i in range(len(demos)):
            new_demos[demo_names[i]] = expert_demos[demo_names[i]]
        return new_demos   
    else:
        return expert_demos

def calculate_q(rewards, dones, gamma=0.98):
    rewards = rewards.astype(np.float32, copy=False)
    dones = dones.astype(np.float32, copy=False)
    q_values = np.zeros_like(rewards)
    for j in range(1, rewards.shape[0]+1):
        q_values[-j] = rewards[-j] + (1- dones[-j]) * gamma * q_values[-j+1]
    
    return q_values

def calculate_q_values(data, gamma=0.98):
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
    for demo_name in list(data.keys()):
        demo = data[demo_name]
        obs = demo['obs']
        next_obs = demo['next_obs']
        actions = demo['actions']
        rewards = demo['rewards']
        dones = demo['dones']
        
        # Check the data sizes:
        assert obs.shape[0] == actions.shape[0] == rewards.shape[0], "Error!, Incompatible sizes"

        q_values = calculate_q(rewards, dones, gamma)
        
        new_data[demo_name] = {'obs':obs, 'next_obs':next_obs, 'actions':actions, 'rewards':rewards, 'dones':dones, 'q_values':q_values}

    return new_data

def calculate_average_sum_rewards(data):
    average_rewards = []
    for demo_name in list(data.keys()):
        demo = data[demo_name]
        rewards = demo['rewards'].astype(np.float32, copy=False)
        average_rewards.append(np.sum(rewards))
    return np.array(average_rewards).mean()

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
    for demo_name in list(data.keys()):
        demo = data[demo_name]
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
    for demo_name in list(data.keys()):
        demo = data[demo_name]
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

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True