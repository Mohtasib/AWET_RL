import os
import yaml
import argparse
import math
import numpy as np
import pandas as pd
import gym
import custom_gym
import time

from awet_rl import AWET_DDPG, AWET_TD3, AWET_SAC
from awet_rl.common.util import listdirs

def test_pendulum(env, model, num_episodes=100, render=False):
    response = []  
    reward = [] 
    success = []
    eps_len = []
    start_time = time.time()
    for _ in range(num_episodes):
        r = 0
        theta = 0
        dist = 0
        succeeded = False
        obs = env.reset()
        for i in range(100):
            action, _states = model.predict(obs)
            obs, new_r, dones, info = env.step(action)
            if render:
                    env.render()
            r += new_r
            if env.reward_type == 'sparse':
                if bool(new_r + 1.0):
                    eps_len.append(i+1)
                    succeeded = True
            else:
                state = math.acos(math.cos(env.state[0]))
                if state <= env.angle_threshold: 
                    eps_len.append(i+1)
                    succeeded = True
            if i >= 90: 
                response.append(state)
                theta += state
        success.append(1) if theta/10.0 <= env.angle_threshold else success.append(0)
        reward.append(r)
        if not succeeded: eps_len.append(100)
    response = np.array(response)
    reward = np.array(reward)
    success = np.array(success)
    eps_len = np.array(eps_len)
    test_time = time.time() - start_time
    return response, reward, success, eps_len, test_time

def test_envs(env, model, num_episodes=100, render=False):
    response = []  
    reward = [] 
    success = []
    eps_len = []
    start_time = time.time()
    for _ in range(num_episodes):
        r = 0
        dist = 0
        succeeded = False
        obs = env.reset()
        for i in range(100):
            action, _states = model.predict(obs)
            obs, new_r, dones, info = env.step(action)
            if render:
                    env.render()
            r += new_r
            if env.reward_type == 'sparse':
                if bool(new_r + 1.0):
                    eps_len.append(i+1)
                    succeeded = True
            else:
                state = -new_r
                if state <= env.distance_threshold: 
                    eps_len.append(i+1)
                    succeeded = True
            if i >= 90: 
                response.append(state)
                dist += state
        success.append(1) if ((dist/10.0) <= env.distance_threshold) else success.append(0)
        reward.append(r)
        if not succeeded: eps_len.append(100)
    response = np.array(response)
    reward = np.array(reward)
    success = np.array(success)
    eps_len = np.array(eps_len)
    test_time = time.time() - start_time
    return response, reward, success, eps_len, test_time

def Test(env_name, exp_path, model_name, num_episodes=100, render=False):
    path = f'{exp_path}/{model_name}/'
    seeds = listdirs(path)
    seeds.sort()

    results_df = pd.DataFrame(columns = ['env_name', 'model_name', 'seed', 'res_mean', 'res_std', 'rew_avg', 'success_rate', 'test_time', 'eps_len'])

    for seed in seeds:
        env = gym.make(env_name)
        if model_name.startswith("AWET_DDPG"): 
            model = AWET_DDPG.load(f'{exp_path}/{model_name}/{seed}/best_model.zip', env=env)
        elif model_name.startswith("AWET_TD3"): 
            model = AWET_TD3.load(f'{exp_path}/{model_name}/{seed}/best_model.zip', env=env)
        elif model_name.startswith("AWET_SAC"):  
            model = AWET_SAC.load(f'{exp_path}/{model_name}/{seed}/best_model.zip', env=env)
        else:
            raise ValueError(f"The agent name must starts with 'AWET_DDPG', 'AWET_TD3', or 'AWET_SAC' and not {model_name}")

        if env_name in ['CustomPendulumDense-v1', 'CustomPendulumSparse-v1']:
            response, reward, success, eps_len, test_time = test_pendulum(env, model, num_episodes, render)
        else:
            response, reward, success, eps_len, test_time = test_envs(env, model, num_episodes, render)
        
        results = [env_name, model_name, seed, round(response.mean(),4), round(response.std(),4), round(reward.mean(),4), round(success.mean(),4), test_time, round(eps_len.mean(),4)]
        print(f'{env_name}, {model_name}, {seed}:')
        print(f'res_mean = {results[3]}, res_std = {results[4]}, rew_avg = {results[5]}, success_rate = {results[6]}, test_time = {results[7]}, eps_len = {results[8]}')
        results_df = results_df.append({'env_name': results[0],
                                        'model_name': results[1],
                                        'seed': results[2],
                                        'res_mean': results[3],
                                        'res_std': results[4],
                                        'rew_avg': results[5],
                                        'success_rate': results[6],
                                        'test_time': results[7],
                                        'eps_len': results[8],
                                        },
                                        ignore_index = True)

    return results_df

def Tester(params):
    print('=========== Testing Started !!!')
    env_name = params['general_params']['env_name']
    exp_path = f"experiments/{params['general_params']['env_name']}/{params['general_params']['exp_name']}"
    models = listdirs(exp_path)

    results_df = pd.DataFrame(columns = ['env_name', 'model_name', 'seed', 'res_mean', 'res_std', 'rew_avg', 'success_rate', 'test_time', 'eps_len'])

    for model in models:
        results_dict = Test(env_name, 
                            exp_path, 
                            model,
                            num_episodes=params['tester_params']['num_episodes'],
                            render=params['tester_params']['render'],
                            )
        results_df = results_df.append(results_dict, ignore_index = True)

    results_df.to_csv(f'{exp_path}/Test_results.csv', index=False)
    # print(Test_results)
    print('=========== Testing Finished !!!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an experiment using AWET algorithm')
    parser.add_argument('--params_path', type=str,   default='configs/pusher/awet_td3.yml', help='parameters directory for training')
    args = parser.parse_args()

    # load paramaters:
    with open(args.params_path) as f:
        params = yaml.safe_load(f)  # params is dict

    Tester(params)
