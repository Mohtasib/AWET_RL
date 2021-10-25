import os
import yaml
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3.common.results_plotter import load_results

from awet_rl.common.util import listdirs

COLORS = [  'tab:orange', 'tab:green', 'tab:blue', 'tab:red', 'tab:pink', 'tab:purple', 'tab:gray',
            'tab:olive', 'tab:cyan', 'k', '#d62728', '#1f77b4', '#7f7f7f', 'tab:brown']

def movingaverage(data, window_size):
    y = np.ones(window_size)
    x = data
    z = np.ones(len(x))
    return np.convolve(x,y,'same') / np.convolve(z,y,'same')

def load_dataset(path, condition, max_num_episodes, smooth_window):
    dirlist = listdirs(path)
    dataset = []
    for subdir in dirlist:
        logs = load_results(path + subdir)
        timesteps = logs['index'].to_numpy()[:max_num_episodes]
        rewards = movingaverage(logs['r'].to_numpy()[:max_num_episodes], smooth_window)
        rewards = rewards[:timesteps.shape[0]]
        data = {'Episodes':timesteps, 'Reward':rewards}
        df = pd.DataFrame(data)
        df.insert(len(df.columns),'Condition',condition)
        dataset.append(df)
    return dataset

def load_all_dataset(path, max_num_episodes=2000, smooth_window=200):
    dirlist = listdirs(path)

    if 'tensorboard_logs' in dirlist:
        dirlist.remove('tensorboard_logs') # Ignore the tensorboard logs folder

    data = []
    for subdir in dirlist:
        data += load_dataset(path + '/' + subdir + '/', subdir, max_num_episodes, smooth_window)

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    return data, dirlist

def Plotter(params):
    print('=========== Plotting Started !!!')
    path = f"experiments/{params['general_params']['env_name']}/{params['general_params']['exp_name']}"

    data, dirlist = load_all_dataset(   path=path, 
                                        max_num_episodes=params['plotter_params']['max_num_episodes'], 
                                        smooth_window=params['plotter_params']['smooth_window'],
                                        )

    fig = plt.figure(   figsize=(params['plotter_params']['width'], params['plotter_params']['height']),
                        dpi=params['plotter_params']['dpi'],
                        )

    palette = dict(zip(dirlist, COLORS))

    sns.set(style="whitegrid", font_scale=params['plotter_params']['font_scale'])
    sns.lineplot(   data=data, x="Episodes", y="Reward", hue='Condition', ci='sd', 
                    linewidth = params['plotter_params']['linewidth'], #legend = False,
                    estimator='mean', palette=palette)
    
    leg = plt.legend(loc='best')
    # plt.legend(loc='upper center', ncol=6, handlelength=1, mode="expand", borderaxespad=0., prop={'size': 20})

    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(params['plotter_params']['linewidth'])

    xscale = np.max(np.asarray(data["Episodes"])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # plt.ylim([-13, -6])
    plt.tight_layout(pad=0.5)

    # plt.legend(loc='best').set_draggable(True)
    # plt.show()
    fig.savefig(f'{path}/Rewards_plot.png')
    print('=========== Plotting Finished !!!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an experiment using AWET algorithm')
    parser.add_argument('--params_path', type=str,   default='configs/pusher/awet_td3.yml', help='parameters directory for training')
    args = parser.parse_args()

    # load paramaters:
    with open(args.params_path) as f:
        params = yaml.safe_load(f)  # params is dict

    Plotter(params)