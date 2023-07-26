import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import *

from gymnasium import Env, spaces
#factorielle = lambda n: n * factorielle(n-1) if n > 0 else 1
import torch
from random import randint
import time
from enum import Enum
from tqdm import trange

#from gym.envs.registration import registry, register, make, spec
from gymnasium.envs.registration import registry, register, make, spec

register(
    id="Simultaneous-v1",                     # Environment ID.
    entry_point="src.simultaneous_env:SimultaneousEnv",  # The entry point for the environment class
    kwargs={
                "n_agents":2, 
                "n_empty":9, 
                "n_ennemies":1, 
                "bonus_win":0.1, 
                "n_agents_to_defeat_ennemy":None                                   # Arguments that go to ForagingEnv's __init__ function.
            },
)

#gymkey = f"Simultaneous-{n_agent}ag-{n_empty}empty-{bonus_win}bonus_win-v0"


class CellEntity(Enum):
    # entity encodings for grid observations
    EMPTY = 0
    ENNEMY = 1

class SimultaneousEnv(Env):
    metadata = {
    "name": "water-bomber-env_v0",
    }

    def __init__(self, n_agents, n_empty, n_ennemies=1, bonus_win=0.1, n_agents_to_defeat_ennemy=None):
        self.n_agents = n_agents
        self.n_empty = n_empty
        self.n_ennemy = n_ennemies
        if n_agents_to_defeat_ennemy is None:
            self.n_agents_to_defeat_ennemy = n_agents
        else:
            self.n_agents_to_defeat_ennemy = n_agents_to_defeat_ennemy
        self.reward_win = 1.0#1.0/(n_empty+1) + bonus_win*((n_empty+1)**(self.n_agents_to_defeat_ennemy))
        self.reward_death = #-1.0/((n_empty+1)**(self.n_agents_to_defeat_ennemy))
        self.common_reward = True

        sa_observation_space = Space(None)
        self.observation_space = spaces.Tuple(tuple(n_agents * [sa_observation_space]))

        sa_action_space = Discrete(n_empty+n_ennemies)
        self.action_space = spaces.Tuple(tuple(n_agents * [sa_action_space]))

        self.render_mode = None
        self.reward_range = (self.reward_death, self.reward_win)

        print("reward_win:",self.reward_win)
        

    def step(self, actions):
        actions = np.array(actions)
        #for a in actions:
        attaquers = actions == 0
        nreward = np.zeros(self.n_agents)
        nreward[attaquers] = self.reward_win if np.sum(attaquers) >= self.n_agents_to_defeat_ennemy else self.reward_death

        nobs = [None for _ in range(self.n_agents)]
        ndone = [True for _ in range(self.n_agents)]
        ninfo = [None for _ in range(self.n_agents)]

        if self.common_reward:
            mean_reward = np.mean(nreward)
            nreward = [mean_reward for _ in range(self.n_agents)]
        return nobs, nreward, ndone, ninfo
    
    def reset(self):
        return [None for _ in range(self.n_agents)]
    
if __name__ == "__main__":
    n_agents=4
    n_empty=10
    env = SimultaneousEnv(n_agents=n_agents, n_empty=n_empty)
    #print("reward:", -1.0/((n_empty+1)**(n_agents-1)))
    total_rewards = np.zeros(n_agents)
    NB_STEPS = 1000000
    nobs = env.reset()
    for _ in trange(NB_STEPS):
        actions = np.random.randint(n_empty+2, size=n_agents)
        nobs, nreward, ndone, ninfo = env.step(actions)
        #print(nreward)
        total_rewards += nreward
    print("Mean reward:", total_rewards/NB_STEPS)
