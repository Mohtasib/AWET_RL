import os

from my_rl.ddpg import DDPG
from my_rl.td3 import TD3
from my_rl.sac import SAC

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file, "r") as file_handler:
    __version__ = file_handler.read().strip()
