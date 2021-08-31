import os
import yaml
import argparse
import gym
import custom_gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.monitor import Monitor

from awet_rl.common.util import SaveOnBestTrainingRewardCallback

def Trainer(params):
    print('=========== Training Started !!!')
    print('=========== Training parameters:')
    print(params)

    save_dir = f"experiments/{params['general_params']['env_name']}/{params['general_params']['exp_name']}/{params['general_params']['agent']}"
    os.makedirs(save_dir, exist_ok=True)

    # Save training parameters:
    with open(f'{save_dir}/params.yml', 'w') as outfile:
        yaml.dump(params, outfile, default_flow_style=False)
    print('=========== Training parameters saved !!!')
    
    for i in range(params['general_params']['num_runs']):
        s = i*10
        print(f'=========== Training Seed_{s}')
        # Create log dir
        log_dir = f"{save_dir}/Seed_{s}"
        os.makedirs(log_dir, exist_ok=True)

        # Create and wrap the environment
        env = gym.make(params['general_params']['env_name'])
        timesteps = env._max_episode_steps * params['general_params']['num_episodes']

        env = Monitor(env, log_dir)

        n_actions = env.action_space.shape[-1]

        if params['general_params']['agent'].startswith('AWET_DDPG'):
            from stable_baselines3.common.noise import NormalActionNoise
            from awet_rl import AWET_DDPG

            # Add some action noise for exploration
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=params['algo_params']['sigma'] * np.ones(n_actions))

            model = AWET_DDPG(  'MlpPolicy', env, action_noise=action_noise, verbose=0, seed=s, 
                                gamma=params['algo_params']['gamma'], 
                                buffer_size=params['algo_params']['buffer_size'], 
                                learning_starts=params['algo_params']['learning_starts'], 
                                gradient_steps=params['algo_params']['gradient_steps'], 
                                learning_rate=params['algo_params']['learning_rate'], 
                                policy_kwargs={'net_arch':params['algo_params']['net_arch'], 'n_critics':params['algo_params']['n_critics']})

        elif params['general_params']['agent'].startswith('AWET_TD3'):
            from stable_baselines3.common.noise import NormalActionNoise
            from awet_rl import AWET_TD3

            # Add some action noise for exploration
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=params['algo_params']['sigma'] * np.ones(n_actions))

            model = AWET_TD3(  'MlpPolicy', env, action_noise=action_noise, verbose=0, seed=s, 
                                gamma=params['algo_params']['gamma'], 
                                buffer_size=params['algo_params']['buffer_size'], 
                                learning_starts=params['algo_params']['learning_starts'], 
                                gradient_steps=params['algo_params']['gradient_steps'], 
                                learning_rate=params['algo_params']['learning_rate'], 
                                policy_kwargs={'net_arch':params['algo_params']['net_arch'], 'n_critics':params['algo_params']['n_critics']})
        
        elif params['general_params']['agent'].startswith('AWET_SAC'):
            from awet_rl import AWET_SAC

            model = AWET_SAC(   'MlpPolicy', env, verbose=0, seed=s, 
                                use_sde=params['algo_params']['use_sde'], 
                                train_freq=params['algo_params']['train_freq'], 
                                gradient_steps=params['algo_params']['gradient_steps'], 
                                learning_rate=params['algo_params']['learning_rate'], 
                                buffer_size=params['algo_params']['buffer_size'], 
                                batch_size=params['algo_params']['batch_size'], 
                                ent_coef=params['algo_params']['ent_coef'],
                                gamma=params['algo_params']['gamma'],
                                tau=params['algo_params']['tau'],
                                policy_kwargs={'log_std_init':params['algo_params']['log_std_init'], 'net_arch':params['algo_params']['net_arch']})

        else:
            raise ValueError(f"The agent name must starts with 'AWET_DDPG', 'AWET_TD3', or 'AWET_SAC' and not {params['general_params']['agent']}")

        # Create the callback: check every 1000 steps
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
        # Train the agent
        model.learn(total_timesteps=int(timesteps), 
                    expert_data_path=params['general_params']['expert_data_path'], 
                    gradient_steps=params['awet_params']['gradient_steps'], 
                    C_e=params['awet_params']['C_e'],
                    C_l=params['awet_params']['C_l'],
                    L_2=params['awet_params']['L_2'],
                    C_clip=params['awet_params']['C_clip'],
                    num_demos=params['awet_params']['num_demos'],
                    AA_mode=params['awet_params']['AA_mode'],
                    ET_mode=params['awet_params']['ET_mode'],
                    callback=callback)

    print('=========== Training Finished !!!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an experiment using AWET algorithm')
    parser.add_argument('--params_path', type=str,   default='configs/pusher/awet_td3.yml', help='parameters directory for training')
    args = parser.parse_args()

    # load paramaters:
    with open(args.params_path) as f:
        params = yaml.safe_load(f)  # params is dict

    Trainer(params)