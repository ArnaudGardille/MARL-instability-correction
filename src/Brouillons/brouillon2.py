from water_bomber_env import WaterBomberEnv

import random
import numpy as np
from time import sleep

scale = 0.25   

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import os
import random
import time
from distutils.util import strtobool
from tqdm import trange
from pprint import pprint

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import Counter
#from stable_baselines3 import DQN

#from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
#from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl

#from smac.env.pettingzoo import StarCraft2PZEnv
from torch.utils.tensorboard import SummaryWriter

#import stable_baselines3 as sb3

from pettingzoo.test import api_test

#from pettingzoo.mpe import simple_v3, simple_spread_v3
#from supersuit import dtype_v0

#import pandas as pd 

import matplotlib.pyplot as plt

from gymnasium.spaces import *

from torchrl.data import TensorDictReplayBuffer
from tensordict import tensorclass, TensorDict
from torchrl.data.replay_buffers.samplers import RandomSampler, PrioritizedSampler
from torchrl.data import LazyTensorStorage, LazyMemmapStorage, ListStorage, TensorDictPrioritizedReplayBuffer

td_target = torch.arange(10).unsqueeze(1).float()
old_val = torch.arange(10, 0, -1).unsqueeze(1).float()

mse_loss = F.mse_loss(td_target, old_val)
#print('mse_loss:', mse_loss)

def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).mean()
mse_loss = weighted_mse_loss(td_target, old_val, torch.ones(10).unsqueeze(1))

#print('weighted_mse_loss:', mse_loss)

x = torch.zeros(10)
n = torch.randint(0, 2, (10,))
m = torch.randint(0, 2, (10,))
x[n==m] = 1.0
x[not n==m] = 2.0
print(x)