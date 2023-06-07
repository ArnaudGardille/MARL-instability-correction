import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import *

from pettingzoo.utils.env import ParallelEnv
from pettingzoo.test import parallel_api_test

factorielle = lambda n: n * factorielle(n-1) if n > 0 else 1
import torch
import time

class WaterBomberEnv(ParallelEnv):
  metadata = {
    "name": "water-bomber-env_v0",
  }

  def __init__(self):

    self.X_MAX = 4
    self.Y_MAX = 4
    self.T_MAX = 100
    self.N_AGENTS = 2

    self.fires = []
    self.water_bombers = []

    self.possible_agents = [0,1]
    #["water_bomber_0", "water_bomber_1"]
    self.verbose = False


  def reset(self, seed=None, options=None):
    self.agents = copy(self.possible_agents)

    self.fires = [[2,3],[4,3]]
    self.water_bombers = [[0,2],[2,2]]

    self.timestep = 0

    observations = self._generate_observations()

    return observations, {}

  def step(self, actions):
    for i, action in actions.items():

      x, y = self.water_bombers[i]

      if action == 0:
        assert y<self.Y_MAX
        self.water_bombers[i][1] += 1
      elif action == 1:
        assert x<self.X_MAX
        self.water_bombers[i][0] += 1
      elif action == 2:
        assert y>0
        self.water_bombers[i][1] -= 1
      elif action == 3:
        assert x>0
        self.water_bombers[i][0] -= 1
    

    terminations = {a: self.water_bombers[a] in self.fires for a in self.agents}

    rewards = {a: self._compute_reward() for a in self.agents}

    truncations = {a: False for a in self.agents}
    if self.timestep > self.T_MAX:
      truncations = {a: True for a in self.agents}
      self.agents = []

    self.timestep += 1


    infos = {a: {} for a in self.agents}
    self.agents = [a for a in self.agents if not (terminations[a] or truncations[a])]
    
    observations = self._generate_observations()

    if self.verbose:
      print()
      print("observations",observations)
      print("rewards",rewards)
      print("terminations",terminations)
      print("truncations",truncations)

    return observations, rewards, terminations, truncations, infos


  def render(self):
    grid = np.full((self.X_MAX+1, self.Y_MAX+1), '_')
    for x, y in self.fires:
      grid[y, x] = "F"

    for i, (x, y) in enumerate(self.water_bombers):
      grid[y, x] = str(i)

    result = "\n".join(["".join([i for i in row]) for row in grid[::-1]])
    print(result)

  @functools.lru_cache(maxsize=None)
  def observation_space(self, agent):
      return Dict({
        'observation': MultiDiscrete(sum([[self.X_MAX+1, self.Y_MAX+1] for _ in range(4)]+[[self.N_AGENTS, self.T_MAX]], [])), #+[13]
        'action_mask': MultiBinary(4)
      })

  @functools.lru_cache(maxsize=None)
  def action_space(self, agent):
      return Discrete(4)


  def _is_terminated(self):
    fires = copy(self.fires)
    water_bombers = copy(self.water_bombers)

    fires.sort()
    water_bombers.sort()

    return fires == water_bombers

  def _compute_reward(self):
    if self._is_terminated():
      return 15.0
      #return 1.0-0.01*factorielle(self.timestep)
    else:
      return -1.0

  def _generate_observations(self):
    action_masks = {}
    for agent in self.agents:
      x, y = self.water_bombers[agent]
      action_mask = np.ones(4)

      if y==self.Y_MAX or [x,y+1] in self.water_bombers:
        action_mask[0] = 0
      if x==self.X_MAX or [x+1,y] in self.water_bombers:
        action_mask[1] = 0
      if y==0 or [x,y-1] in self.water_bombers:
        action_mask[2] = 0
      if x==0 or [x-1,y] in self.water_bombers:
        action_mask[3] = 0

      action_masks[agent] = action_mask

    observations = {
      a: {
        "observation":torch.flatten(torch.tensor(self.fires + self.water_bombers +[[a, self.timestep]], dtype=torch.float)), #+ [self.timestep]
        "action_mask":action_masks[a]
      } 
      for a in self.agents
    }
    return observations


if __name__ == "__main__":
  env = WaterBomberEnv()
  parallel_api_test(env, num_cycles=1_000_000)

  observations, _ = env.reset(seed=42)
  env.render()
  #print("observations initiale:", observations)
  while env.agents:
    # this is where you would insert your policy
    actions = {agent: np.random.choice(np.nonzero(observations[agent]['action_mask'])[0]) for agent in env.agents}  

    #actions = {agent: env.action_space(agent).sample() for agent in env.agents}  
    #print("actions:",actions)
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print("rewards:",rewards)

    env.render()
    time.sleep(0.2)
  env.close()
