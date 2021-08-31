
import os
import yaml
import gym
import custom_gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.monitor import Monitor

from awet_rl.common.util import SaveOnBestTrainingRewardCallback

from stable_baselines3.common.noise import NormalActionNoise
from awet_rl import AWET_TD3

params = {  
            # General parameters:
            'general_params' : {'exp_name' : 'Algo_Exp',
                                'agent' : 'AWET_TD3',
                                'env_name' : 'CustomPusherDense-v1',
                                'expert_data_path' : 'demos_data/pusher/Pusher_Dense_Demos_100.pkl',
                                'num_episodes' : 2000,
                                'num_runs' : 10,
            },

            # AWET parameters:
            'awet_params' : {   'C_e' : 0.60,
                                'C_l' : 0.80,
                                'L_2' : 0.005,
                                'C_clip' : 1.6,
                                'num_demos' : 100,
                                'AA_mode' : True,
                                'ET_mode' : True,
                                'gradient_steps' : 1000,
            },

            # TD3 parameters:
            'algo_params' : {   'sigma' : 0.05,
                                'gamma' : 0.98,
                                'buffer_size' : 200000,
                                'learning_starts' : 0,
                                'gradient_steps' : -1,
                                'learning_rate' : float(1e-3),
                                'net_arch' : [400, 300],
                                'n_critics' : 2,
            },

            # Plotter parameters:
            'plotter_params' : {'max_num_episodes' : 2000,
                                'smooth_window' : 200,
                                'width' : 10,
                                'height' : 8,
                                'dpi' : 70,
                                'font_scale' : 2.5,
                                'linewidth' : 3.5,
            },

            # Tester parameters:
            'tester_params' : { 'num_episodes' : 100,
                                'render' : False,
            },
        }

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

    # Add some action noise for exploration
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=params['algo_params']['sigma'] * np.ones(n_actions))

    model = AWET_TD3(  'MlpPolicy', env, action_noise=action_noise, verbose=0, seed=s, 
                        gamma=params['algo_params']['gamma'], 
                        buffer_size=params['algo_params']['buffer_size'], 
                        learning_starts=params['algo_params']['learning_starts'], 
                        gradient_steps=params['algo_params']['gradient_steps'], 
                        learning_rate=params['algo_params']['learning_rate'], 
                        policy_kwargs={'net_arch':params['algo_params']['net_arch'], 'n_critics':params['algo_params']['n_critics']})

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