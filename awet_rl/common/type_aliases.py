"""Common aliases for type hints"""

from enum import Enum
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union

import gym
import numpy as np
import torch as th


class ExtendedReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    extras: th.Tensor