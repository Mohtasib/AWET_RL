import os

from awet_rl.ddpg import AWET_DDPG
from awet_rl.td3 import AWET_TD3
from awet_rl.sac import AWET_SAC

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file, "r") as file_handler:
    __version__ = file_handler.read().strip()
