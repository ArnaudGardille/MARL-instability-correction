import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import *

from sklearn.preprocessing import OneHotEncoder
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.test import parallel_api_test

#factorielle = lambda n: n * factorielle(n-1) if n > 0 else 1
import torch
from random import randint
import time

class WaterBomberEnv(ParallelEnv):
  metadata = {
    "name": "water-bomber-env_v0",
  }

  def __init__(self, x_max=4, y_max=4, t_max=20, n_agents=2, add_id=False):

    self.X_MAX = x_max
    self.Y_MAX = y_max
    self.T_MAX = t_max
    self.N_AGENTS = n_agents


    self.possible_agents = ["water_bomber_"+str(i) for i in range(n_agents)]
    self.symbols = {"water_bomber_"+str(i):str(i) for i in range(n_agents)}
    self.verbose = False

    self.add_id = add_id
    self.length_id = self.N_AGENTS if add_id else 0
    if self.add_id:
      enc = OneHotEncoder(sparse_output=False).fit(np.array(self.possible_agents).reshape(-1, 1))
      self.one_hot = {agent:enc.transform(np.array([agent]).reshape(-1, 1))[0] for agent in self.possible_agents}


  def reset(self, seed=None, options=None, deterministic=False):
    self.agents = copy(self.possible_agents)

    if deterministic:
      assert self.X_MAX==4 and self.Y_MAX==4 and self.T_MAX==20 and self.N_AGENTS==2
      self.fires = [[2,3],[4,3]]
      self.water_bombers = {"water_bomber_0":[0,2],"water_bomber_1":[2,2]}
    else:
      points = {(randint(0, self.X_MAX), randint(0, self.Y_MAX))}
      while len(points) < 2*self.N_AGENTS:
        points |= {(randint(0, self.X_MAX), randint(0, self.Y_MAX))}
      list_pos = list(list(x) for x in points)
      
      self.fires = list_pos[:self.N_AGENTS]
      self.water_bombers = {"water_bomber_"+str(i):coor for i, coor in enumerate(list_pos[self.N_AGENTS:])}

    self.has_finished = [False]*self.N_AGENTS

    self.timestep = 0

    observations = self._generate_observations()

    reward_opti = self.compute_optimal_reward()
    infos = {a: {'reward_opti':reward_opti} for a in self.agents}

    return observations, infos

  def step(self, actions):
    for agent, action in actions.items():

      x, y = self.water_bombers[agent]

      if action == 0:
        assert y<self.Y_MAX
        self.water_bombers[agent][1] += 1
      elif action == 1:
        assert x<self.X_MAX
        self.water_bombers[agent][0] += 1
      elif action == 2:
        assert y>0
        self.water_bombers[agent][1] -= 1
      elif action == 3:
        assert x>0
        self.water_bombers[agent][0] -= 1
    

    #terminations = {a: self.water_bombers[a] in self.fires for a in self.agents}

    self.has_finished = {a:self.water_bombers[a] in self.fires for a in self.agents}

    rewards = {a: self._compute_reward() for a in self.agents}

    self.timestep += 1

    infos = {a: {} for a in self.agents}

    terminations = {a: False for a in self.agents}
    #if np.all(self.has_finished):
    #  terminations = {a: True for a in self.agents}
      #self.agents = []

    truncations = {a: False for a in self.agents}
    observations = self._generate_observations()
    
    if self.timestep > self.T_MAX:
      truncations = {a: True for a in self.agents}
      self.agents = []

    #self.agents = [a for a in self.agents if not (terminations[a] or truncations[a])]
    

    if self.verbose:
      print()
      print("observations",observations)
      print("rewards",rewards)
      print("terminations",terminations)
      print("truncations",truncations)

    return observations, rewards, terminations, truncations, infos


  def render(self):
    grid = np.full((self.Y_MAX+1, self.X_MAX+1), '_')
    for x, y in self.fires:
      grid[y, x] = "F"

    for agent, (x, y) in self.water_bombers.items():
      grid[y, x] = self.symbols[agent]

    result = "\n".join(["".join([i for i in row]) for row in grid[::-1]])
    print()
    print(result)

  @functools.lru_cache(maxsize=None)
  def observation_space(self, agent):
    l = sum([[self.X_MAX+1, self.Y_MAX+1] for _ in range(2*self.N_AGENTS)]+[[self.T_MAX]], [])
    if self.add_id:
      l += self.length_id*[2]
    return Dict({
      'observation': MultiDiscrete(l), #+[13]
      'action_mask': MultiBinary(5)
    })

  @functools.lru_cache(maxsize=None)
  def action_space(self, agent):
      return Discrete(5)


  def _is_terminated(self):
    fires = copy(self.fires)
    water_bombers = copy(list(self.water_bombers.values()))

    fires.sort()
    water_bombers.sort()

    return fires == water_bombers

  def _compute_reward(self):
    #if self._is_terminated():
    if np.all([self.has_finished[a] for a in self.agents]):
      return 1.0 #15.0
      #return 1.0-0.01*factorielle(self.timestep)
    else:
      return 0.0 #-1.0

  def get_action_mask(self, x, y):
    #x, y = self.water_bombers[]

    action_mask = np.ones(5)
    if [x,y] in self.fires: #self.has_finished[agent]:
      action_mask = np.array([0,0,0,0,1])
    else:
      action_mask[-1] = 0

    occupied_positions = list(self.water_bombers.values())
    if y==self.Y_MAX or [x,y+1] in occupied_positions:
      action_mask[0] = 0
    if x==self.X_MAX or [x+1,y] in occupied_positions:
      action_mask[1] = 0
    if y==0 or [x,y-1] in occupied_positions:
      action_mask[2] = 0
    if x==0 or [x-1,y] in occupied_positions:
      action_mask[3] = 0

    return action_mask

  def normalize_obs(self, obs):
    # ASSUMES ALL AGENTS HAVE SAME OBS SPACE
    normalized_obs = copy(obs)
    normalized_obs['observation']

    agent = self.possible_agents[0]
    normalized_obs['observation'] = 2*normalized_obs['observation'].cpu()/(self.observation_space(agent)['observation'].nvec-1) - 1.0
    normalized_obs['observation'] = normalized_obs['observation'].float()
    return normalized_obs

  def _generate_observations(self):
    action_masks = {}
    for agent in self.agents:
      x, y = self.water_bombers[agent]

      action_masks[agent] = self.get_action_mask(x,y)

    occupied_positions = list(self.water_bombers.values())

    obs = sum(self.fires + occupied_positions +[[self.timestep]], [])
    observations = {}
    for a in self.agents:
      obs_perso = np.concatenate((obs, self.one_hot[a])) if self.add_id else obs
      observations[a] = {
        "observation":torch.tensor(obs_perso, dtype=torch.float), #+ [self.timestep]
        "action_mask":torch.tensor(action_masks[a], dtype=torch.float)
      }

    return observations

  def compute_optimal_reward(self):
    norm_1 = lambda x, y: np.linalg.norm(np.array(x, dtype=float)-np.array(y, dtype=float), ord=1)
    duree_min = min([max([norm_1(self.water_bombers[a],f) for f in self.fires]) for a in self.agents])
    return self.T_MAX - duree_min + 2.0


if __name__ == "__main__":
  env = WaterBomberEnv(x_max=3, y_max=3, t_max=20, n_agents=3, deterministic=False, add_id=True)
  parallel_api_test(env, num_cycles=1_000_000)

  observations, infos = env.reset(seed=42)
  print('infos:', infos)
  env.render()
  #print("observations initiale:", observations)
  total_reward = 0.0
  while env.agents:
    # this is where you would insert your policy
    actions = {agent: np.random.choice(np.nonzero(observations[agent]['action_mask'])[0]) for agent in env.agents}  

    #actions = {agent: env.action_space(agent).sample() for agent in env.agents}  
    #print("actions:",actions)
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(observations)
    total_reward += np.mean(list(rewards.values())) 
    print("rewards:",rewards, "; total reward:", total_reward)

    env.render()
    time.sleep(0.1)
  env.close()
