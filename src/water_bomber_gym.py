import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import *

from gymnasium import Env
#factorielle = lambda n: n * factorielle(n-1) if n > 0 else 1
import torch
from random import randint
import time
from enum import Enum

class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4


class CellEntity(Enum):
    # entity encodings for grid observations
    OUT_OF_BOUNDS = 0
    EMPTY = 1
    FOOD = 2
    AGENT = 3

class WaterBomberEnv(Env):
  metadata = {
    "name": "water-bomber-env_v0",
  }

  def __init__(self, x_max=4, y_max=4, t_max=20, n_agents=2, add_id=False, obs_normalization=True, deterministic=False, gamma=0.9):

    self.X_MAX = x_max
    self.Y_MAX = y_max
    self.T_MAX = t_max
    self.n_agents = n_agents
    self.obs_normalization = obs_normalization
    self.deterministic = deterministic
    self.gamma=gamma

    #self.players = [Player() for _ in range(n_agents)]
    self.possible_agents = [i for i in range(n_agents)]
    self.name_agents = ["water_bomber_"+str(i) for i in range(n_agents)]
    self.symbols = {i:str(i) for i in range(n_agents)}
    #{"water_bomber_"+str(i):str(i) for i in range(n_agents)}
    self.verbose = False

    self.add_id = add_id
    self.length_id = self.n_agents if add_id else 0
    #if self.add_id:
      #enc = OneHotEncoder(sparse_output=False).fit(np.array(self.possible_agents).reshape(-1, 1))
      #self.one_hot = {agent:enc.transform(np.array([agent]).reshape(-1, 1))[0] for agent in self.possible_agents}
    self.one_hot = np.eye(n_agents)

  def reset(self, seed=None, options=None, deterministic=False):
    self.agents = copy(self.possible_agents)

    if self.deterministic or deterministic:
      assert self.X_MAX==4 and self.Y_MAX==2 and self.T_MAX==20 and self.n_agents==2, (self.X_MAX==4, self.Y_MAX==4, self.T_MAX==20, self.n_agents==2)
      self.fires = [[2,1],[4,1]]
      self.water_bombers = [[0,0],[2,0]]
    else:
      points = {(randint(0, self.X_MAX), randint(0, self.Y_MAX))}
      while len(points) < 2*self.n_agents:
        points |= {(randint(0, self.X_MAX), randint(0, self.Y_MAX))}
      list_pos = list(list(x) for x in points)
      
      self.fires = list_pos[:self.n_agents]
      self.water_bombers = list_pos[self.n_agents:]
      #self.water_bombers = {"water_bomber_"+str(i):coor for i, coor in enumerate(list_pos[self.n_agents:])}

    self.has_finished = [False]*self.n_agents

    self.timestep = 0

    observations, action_masks = self._generate_observations()

    self.reward_opti = self.compute_optimal_reward()
    #infos = {a: {'reward_opti':reward_opti} for a in self.agents}

    return observations, action_masks

  def step(self, actions):
    for agent, action in enumerate(actions):

      x, y = self.water_bombers[agent]

      occupied_positions = self.water_bombers
      #list(self.water_bombers.values())

      if action == 0 and not [x,y+1] in occupied_positions:
        assert y<self.Y_MAX
        self.water_bombers[agent][1] += 1
      elif action == 1 and not [x+1,y] in occupied_positions:
        assert x<self.X_MAX
        self.water_bombers[agent][0] += 1
      elif action == 2 and not [x,y-1] in occupied_positions:
        assert y>0
        self.water_bombers[agent][1] -= 1
      elif action == 3 and not [x-1,y] in occupied_positions:
        assert x>0
        self.water_bombers[agent][0] -= 1
    

    #terminations = {a: self.water_bombers[a] in self.fires for a in self.agents}

    self.has_finished = {a:self.water_bombers[a] in self.fires for a in self.agents}

    rewards = [self._compute_reward() for a in self.agents]
    #{a: self._compute_reward() for a in self.agents}

    self.timestep += 1

    infos = [{} for a in self.agents]

    #terminations = [False for a in self.agents]
    #if np.all(self.has_finished):
    #  terminations = {a: True for a in self.agents}
      #self.agents = []

    truncations = [False for a in self.agents]
    observations, action_masks = self._generate_observations()
    
    if self.timestep > self.T_MAX:
      truncations = [True for a in self.agents]
      #self.agents = []

    #self.agents = [a for a in self.agents if not (terminations[a] or truncations[a])]
    

    if self.verbose:
      print()
      print("observations",observations)
      print("rewards",rewards)
      #print("terminations",terminations)
      print("truncations",truncations)

    return observations, rewards, truncations, action_masks


  def render(self):
    grid = np.full((self.Y_MAX+1, self.X_MAX+1), '_')
    for x, y in self.fires:
      grid[y, x] = "F"

    for agent, (x, y) in enumerate(self.water_bombers): #.items():
      grid[y, x] = self.symbols[agent]

    result = "\n".join(["".join([i for i in row]) for row in grid[::-1]])
    print()
    print(result)

  @functools.lru_cache(maxsize=None)
  def observation_space(self, agent):
    l = sum([[self.X_MAX+1, self.Y_MAX+1] for _ in range(2*self.n_agents)]+[[self.T_MAX]], []) #
    if self.add_id:
      l += self.length_id*[2]
    return  MultiDiscrete(l)
    #return Dict({
    #  'observation': MultiDiscrete(l), #+[13]
    #  'action_mask': MultiBinary(5)
    #})

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
      return 1.0/self.reward_opti #15.0
      #return 1.0-0.01*factorielle(self.timestep)
    else:
      return 0.0 #-1.0

  def get_action_mask_from_id(self, agent_id):
    return self.get_action_mask(*self.water_bombers[agent_id])

  def get_action_mask(self, x, y):
    #x, y = self.water_bombers[]

    if [x,y] in self.fires: #self.has_finished[agent]:
      return np.array([0,0,0,0,1])
    
    action_mask = np.ones(5)
    action_mask[-1] = 0

    occupied_positions = self.water_bombers
    #list(self.water_bombers.values())
    if y==self.Y_MAX or [x,y+1] in occupied_positions:
      action_mask[0] = 0
    if x==self.X_MAX or [x+1,y] in occupied_positions:
      action_mask[1] = 0
    if y==0 or [x,y-1] in occupied_positions:
      action_mask[2] = 0
    if x==0 or [x-1,y] in occupied_positions:
      action_mask[3] = 0

    if sum(action_mask) == 0:
      return np.array([0,0,0,0,1])

    #assert sum(action_mask) > 0, (self.render(), action_mask, self.fires, occupied_positions, x, y)
    return action_mask

  #def get_avail_agent_actions(self):

  def normalize_obs(self, obs):
    # ASSUMES ALL AGENTS HAVE SAME OBS SPACE
    normalized_obs = copy(obs)

    agent = self.possible_agents[0]
    normalized_obs = 2*normalized_obs.cpu()/(self.observation_space(agent).nvec-1) - 1.0
    normalized_obs = normalized_obs.float()
    #assert sum(normalized_obs['action_mask']) > 0
    return normalized_obs

  def get_state_and_action_masks(self):
    action_masks = [] #{}

    for agent in self.agents:
      x, y = self.water_bombers[agent]

      action_masks.append(self.get_action_mask(x,y))

    occupied_positions = self.water_bombers
    #list(self.water_bombers.values())

    state = sum(self.fires + occupied_positions +[[self.timestep]], [])
    return state, action_masks

  def _generate_observations(self):
     #+[[self.timestep]]
    state, action_masks = self.get_state_and_action_masks()
    observations = []
    for a in self.agents:
      obs_perso = np.concatenate((state, self.one_hot[a])) if self.add_id else state
      obs_perso = torch.tensor(obs_perso, dtype=torch.float)
      if self.obs_normalization:
        obs_perso = self.normalize_obs(obs_perso)
      observations.append(obs_perso)
      action_masks[a] = torch.tensor(action_masks[a], dtype=torch.float)
      """observations[a] = {
        "observation":torch.tensor(obs_perso, dtype=torch.float), #+ [self.timestep]
        "action_mask":torch.tensor(action_masks[a], dtype=torch.float)
      }"""

    return observations, action_masks

  def compute_optimal_reward(self):
    norm_1 = lambda x, y: np.linalg.norm(np.array(x, dtype=float)-np.array(y, dtype=float), ord=1)
    duree_min = min([max([norm_1(self.water_bombers[a],f) for f in self.fires]) for a in self.agents])
    return self.T_MAX - duree_min + 2.0
    #return (1-self.gamma**(N))/(1-self.gamma)

  def get_state(self):
    return self.get_state_and_action_masks()[0]

  def get_env_info(self):
    return {
      "n_actions": 4,
      "n_agents": self.n_agents,
    }

def main_1():
  env = WaterBomberEnv(x_max=4, y_max=1, t_max=20, n_agents=2, add_id=True)
  #parallel_api_test(env, num_cycles=1_000_000)
  observations, action_masks = env.reset(seed=42, deterministic=False, )
  print('action_masks:', action_masks)
  #env.render()
  #print("observations initiale:", observations)
  total_reward = 0.0
  done = False
  while not done:
    # this is where you would insert your policy
    #actions = {agent: np.random.choice(np.nonzero(observations[agent]['action_mask'])[0]) for agent in env.agents}  
    #print(observations)
    actions = [np.random.choice(np.nonzero(action_masks[agent])[0]) for agent in env.agents]

    #actions = {agent: env.action_space(agent).sample() for agent in env.agents}  
    #print("actions:",actions)
    env.render()
    print("actions", actions)
    observations, rewards, terminations, action_masks = env.step(actions)
    print("observations", observations)
    print("rewards", rewards)
    print("terminations", terminations)
    print("action_masks", action_masks)
    print()
    done = np.all(np.array(terminations)==True)
    total_reward += np.mean(rewards) 
    #print(actions, observations, rewards, terminations, action_masks)
    #print("rewards:",rewards, "; total reward:", total_reward)

    #env.render()
    time.sleep(0.1)
  env.close()

def main_2():
  envs = [WaterBomberEnv(x_max=3, y_max=3, t_max=20, n_agents=3, add_id=True) for _ in range(10)]
  #for i in range(10):
  observations = [envs[i].reset(seed=i, deterministic=False, )[0] for i in range(10)]
  #env.render()
  #print("observations initiale:", observations)
  total_reward = [0.0 for i in range(10)]
  print([envs[i].agents for i in range(10)])
  while np.any([envs[i].agents != [] for i in range(10)]):
    for i in range(10):
      if envs[i].agents:
        # this is where you would insert your policy
        actions = {agent: np.random.choice(np.nonzero(observations[i][agent]['action_mask'])[0]) for agent in envs[i].agents}  

        #actions = {agent: env.action_space(agent).sample() for agent in env.agents}  
        #print("actions:",actions)
        observations[i], rewards, terminations, truncations, infos = envs[i].step(actions)
        #print(observations)
        total_reward += np.mean(list(rewards.values())) 
        print("rewards:",rewards, "; total reward:", total_reward)

        #env.render()
        #time.sleep(0.1)
      #else:
        #env.close()
if __name__ == "__main__":
  main_1()
