import os
import argparse
import yaml

from awet_rl.core import Trainer, Plotter, Tester

def run_job(params):
    print('=========== Job Started !!!')
    Trainer(params)
    Plotter(params)
    Tester(params)
    print('=========== Job Finished !!!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an experiment using AWET algorithm')
    parser.add_argument('--params_path', type=str,   default='configs/pusher/awet_td3.yml', help='parameters directory for training')
    args = parser.parse_args()

    # load paramaters:
    with open(args.params_path) as f:
        params = yaml.safe_load(f)  # params is dict

    run_job(params)