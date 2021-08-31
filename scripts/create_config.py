import os
import yaml

ddpg_params = {  
            # General parameters:
            'general_params' : {'exp_name' : 'Algo_Exp',
                                'agent' : 'AWET_DDPG',
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

            # DDPG parameters:
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

td3_params = {  
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

sac_params = {  
            # General parameters:
            'general_params' : {'exp_name' : 'Algo_Exp',
                                'agent' : 'AWET_SAC',
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

            # SAC parameters:
            'algo_params' : {   'use_sde' : True, 
                                'train_freq' : 64, 
                                'gradient_steps' : 64, 
                                'learning_rate' : float(7.3e-4), 
                                'buffer_size' : 300000, 
                                'batch_size' : 256, 
                                'ent_coef' : 'auto',
                                'gamma' : 0.98,
                                'tau' : 0.02,
                                'log_std_init' : -3,
                                'net_arch' : [64, 64],
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


params = ddpg_params
save_dir = './configs/pusher'
os.makedirs(save_dir, exist_ok=True)

file_path = f'{save_dir}/awet_td3.yml'

with open(file_path, 'w') as outfile:
    yaml.dump(params, outfile, default_flow_style=False)


