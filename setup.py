from os.path import join, dirname, realpath
from setuptools import setup

def read_requirements_file(filename):
    req_file_path = '%s/%s' % (dirname(realpath(__file__)), filename)
    with open(req_file_path) as f:
        return [line.strip() for line in f]

setup(name='awet_rl',
      version='1.0.0',
      description="This is the implementation of AWET algorithm for DDPG, TD3, and SAC and it is based on the stable_baselines3",
      author="Abdalkarim Mohtasib",
      author_email='amohtasib@lincoln.ac.uk',
      platforms=["any"],
      license="GPLv3",
      url="https://github.com/Mohtasib/AWET_RL",
      install_requires=read_requirements_file('requirements.txt'),
      python_requires='>=3.6',
)
