


from myenvs.simultaneous_attack import *
from myenvs.water_bomber import *
from torch.distributions.categorical import Categorical
import torchsnapshot
import shutil

from gym import Wrapper, ObservationWrapper
from gym.spaces import MultiDiscrete, Box
import datetime

import random
import yaml
import numpy as np
from time import sleep
from copy import deepcopy
import pickle
import pandas as pd 
import sys
import wandb
import argparse
import os
from pathlib import Path
import random
import time
from distutils.util import strtobool
from tqdm import trange
from pprint import pprint
from copy import copy
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pygame
from collections import Counter
from torch.nn.functional import sigmoid

#from smac.env.pettingzoo import StarCraft2PZEnv
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from gym.spaces import *

# For replay buffer
from tensordict import tensorclass, TensorDict
from torchrl.data.replay_buffers.samplers import RandomSampler, PrioritizedSampler
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage, LazyMemmapStorage, ListStorage, TensorDictPrioritizedReplayBuffer

# Warnings supression
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("error", category=RuntimeWarning)
scale = 0.25   
acting_device = 'cpu' #GPU is only used for training

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    #parser.add_argument("--correct-prio-small-buffer", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #parser.add_argument("--correct-prio-big-buffer", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    parser.add_argument("--correct-prio", type=lambda x: bool(strtobool(x)), nargs="?", const=True,
        help="")
    parser.add_argument("--device", type=str, choices=['cpu', 'mps', 'cuda'], nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--plot-q-values", type=lambda x: bool(strtobool(x)), nargs="?", const=True,
        help="whether to plot the q values. only available for the water-bomber env")
    parser.add_argument("--visualisation", type=lambda x: bool(strtobool(x)), nargs="?", const=True, default=False,
        help="Render the environment. Agents won't be trained.")
    parser.add_argument("--use-state", type=lambda x: bool(strtobool(x)), nargs="?", const=True,
        help="whether we give the global state to agents instead of their respective observation")
    parser.add_argument("--save-buffer", type=lambda x: bool(strtobool(x)), nargs="?", const=True,
        help="saves the replay buffer inside the experiment folder")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--last-buffer", type=lambda x: bool(strtobool(x)), nargs="?", const=True)
    parser.add_argument("--fixed-buffer", type=lambda x: bool(strtobool(x)), nargs="?", const=True,
        help="Nothing will be added to the buffer(only useful if it has been loaded)")
    parser.add_argument("--buffer-on-disk", type=lambda x: bool(strtobool(x)), nargs="?", const=True,
        help="The buffer will be stored on disk. Useful if it is to big to fit in the RAM.")
    

    # Environment specific arguments
    parser.add_argument("--enforce-coop", type=lambda x: bool(strtobool(x)), nargs="?", const=True,
        help="Coop version for lbf, and mix up rewards for simultaneous")
    parser.add_argument("--env-id", choices=['simultaneous', 'water-bomber', 'smac', 'lbf', 'mpe'] ,default='simultaneous',
        help="the id of the environment")
    parser.add_argument("--x-max", type=int, help="Only for the water-bomber env")
    parser.add_argument("--y-max", type=int, help="Only for the water-bomber env")
    parser.add_argument("--t-max", type=int, help="Maximum episode duration")
    parser.add_argument("--n-agents", type=int)
    parser.add_argument("--env-normalization", type=lambda x: bool(strtobool(x)), nargs="?", const=True)
    parser.add_argument("--map", type=str, nargs="?", const=True, 
        help="Select the map.  For SMAC: ['10m_vs_11m', '27m_vs_30m', '2c_vs_64zg', '2s3z', '2s_vs_1sc', '3s5z', '3s5z_vs_3s6z', '3s_vs_5z', 'bane_vs_bane', 'corridor', 'MMM', 'MMM2']. For MPE: ['adversary', 'crypto', 'push', 'reference', 'speaker_listener', 'spread', 'tag', 'world_comm', 'simple']")
    # Algorithm specific arguments
    parser.add_argument("--load-agents-from", type=str, default=None,
        help="the experiment from which to load agents.")
    parser.add_argument("--load-buffer-from", type=str, default=None,
        help="the experiment from which to load agents.")
    parser.add_argument("--random-policy", type=lambda x: bool(strtobool(x)), nargs="?", const=True)
    parser.add_argument("--total-timesteps", type=int, 
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, 
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, 
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float,
        help="the target network update rate")
    parser.add_argument("--evaluation-frequency", type=int, 
        help="number of episodes between two evaluations")
    parser.add_argument("--evaluation-episodes", type=int, 
        help="number of evaluation episodes that will be averaged")
    parser.add_argument("--target-network-frequency", type=int, 
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int,  
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, 
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, 
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, 
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, 
        help="timestep to start learning")
    parser.add_argument("--n-actions", type=int)
    parser.add_argument("--train-frequency", type=int, 
        help="the frequency of training")
    parser.add_argument("--single-agent", type=lambda x: bool(strtobool(x)), nargs="?", const=True,
        help="whether to use a single network for all agents. Identity is the added to observation")
    parser.add_argument("--add-id", type=lambda x: bool(strtobool(x)), nargs="?", const=True,
        help="whether to add agents identity to observation")
    parser.add_argument("--add-epsilon", type=lambda x: bool(strtobool(x)), nargs="?", const=True,
        help="whether to add epsilon to observation")
    parser.add_argument("--prioritize-big-buffer", type=lambda x: bool(strtobool(x)), nargs="?", const=True
        ,help="Wheter to do prioritized experience replay on the big replay buffer (when using likely of laber)")
    parser.add_argument("--dueling", type=lambda x: bool(strtobool(x)), nargs="?", const=True,
        help="whether to use a dueling network architecture.")
    parser.add_argument("--deterministic-env", type=lambda x: bool(strtobool(x)), nargs="?", const=True)
    parser.add_argument("--prio", choices=['none', 'td_error', 'td-past', 'td-cur-past', 'td-cur', 'cur-past', 'cur', 'past'], nargs="?", const=True
        ,help="Select the priorisation quantity for PER of Laber")
    parser.add_argument("--filter", choices=['none', 'td_error', 'td-past', 'td-cur-past', 'td-cur', 'cur-past', 'cur', 'past'], nargs="?", const=True
        ,help="Select the priorisation quantity for Likely")
    parser.add_argument("--loss-correction-for-others", choices=['none','td_error', 'td-past', 'td-cur-past', 'td-cur', 'cur-past', 'cur'], default=None
        ,help="How to correct the loss for the others evolution. please refer to the article for more explanations")
    parser.add_argument("--correction-modification", choices=['none', 'sqrt', 'sigmoid', 'normalize'] , nargs="*"
        ,help="function that will be applied to the loss correction")
    parser.add_argument("--clip-correction-after", type=float, nargs="?", const=True
        ,help="Allows to clip the correction")
    parser.add_argument("--rb", choices=['uniform', 'prioritized', 'laber', 'likely', 'correction'], 
        help="whether to use a uniform, prioritized, Laber of Likely replay buffer.")
    parser.add_argument("--add-others-explo", type=lambda x: bool(strtobool(x)), nargs="?", const=True
        ,help="whether to add a boolean vector of wheter each other agent is exploring to observation")
    args = parser.parse_args()

    return args


def weighted_mse_loss(predicted, target, weight):
    return (weight * (predicted - target) ** 2).mean()

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module): #QNetworkSimpleMLP
    def __init__(self, obs_shape, act_shape):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, act_shape),
        )

    def forward(self, x):
        return self.network(x)

class DuelingQNetwork(nn.Module):
    def __init__(self, obs_shape, act_shape):
        super(DuelingQNetwork, self).__init__()

        self.fc1 = nn.Linear(obs_shape, 64)
        self.relu = nn.ReLU()
        self.fc_value = nn.Linear(64, 256)
        self.fc_adv = nn.Linear(64, 256)

        self.value = nn.Linear(256, 1)
        self.adv = nn.Linear(256, act_shape)

    def forward(self, state, value_only=False):
        y = self.relu(self.fc1(state))
        value = self.relu(self.fc_value(y))
        adv = self.relu(self.fc_adv(y))

        value = self.value(value)
        if value_only:
            return value

        adv = self.adv(adv)

        advAverage = torch.mean(adv, dim=-1, keepdim=True)
        Q = value + adv - advAverage

        return Q

    def select_action(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_index = torch.argmax(Q, dim=1)
        return action_index.item()

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def create_rb(rb_type, buffer_size, batch_size, n_agents, device, prio, prioritize_big_buffer=False, path=None):
    smaller_buffer = None
    if path is not None:
        os.makedirs(path, exist_ok=True)
        shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        print("Replay buffer location:", path/'replay_buffer')
        rb_storage = LazyMemmapStorage(buffer_size, device=device, scratch_dir=path/'replay_buffer')
    else:
        rb_storage = LazyTensorStorage(buffer_size, device=device)
    if rb_type == 'uniform' or rb_type == 'correction':
        replay_buffer = TensorDictReplayBuffer(
            storage=rb_storage,
            batch_size=batch_size,
        )

    elif rb_type == 'prioritized':
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha = 1.0,
            beta = 1.0,
            priority_key="td_error",
            storage=rb_storage,
            batch_size=batch_size,
        )
        
    else:
        if prioritize_big_buffer:
            replay_buffer = TensorDictPrioritizedReplayBuffer(
                storage=rb_storage,
                alpha = 1.0,
                beta = 1.0,
                priority_key="td_error",
                batch_size=10*batch_size,
            )
        else:
            replay_buffer = TensorDictReplayBuffer(
                storage=rb_storage,
                #priority_key="td_error",
                batch_size=10*batch_size,
            )
        

        if rb_type =='laber':
            smaller_buffer_size = 10*batch_size

            smaller_buffer = TensorDictPrioritizedReplayBuffer(
                alpha = 1.0,
                beta = 1.0,
                priority_key=prio,
                storage=LazyTensorStorage(smaller_buffer_size),
                batch_size=batch_size,
            )

    return replay_buffer, smaller_buffer


def current_and_past_others_actions_likelyhood(sample, agents, epsilon, single_agent):
        current_likelyhood, past_likelyhood = [], []
        for agent_id, agent in enumerate(agents):
            
            sample = sample.to(agent.device)
            obs = sample['observations'][:, agent_id].float()
            action_mask = sample['action_mask'][:, agent_id]
            next_obs = sample['next_observations'][:, agent_id].float()
            next_action_mask = sample['next_action_mask'][:, agent_id]
            reward = sample['rewards'][:, agent_id]
            dones = sample['dones'][:, agent_id]
            actions = sample['actions'][:, agent_id]
            old_probability = sample['actions_likelihood'][:,agent_id]

            if agent.params['add_id']: 
                batch_id = agent.one_hot_id.repeat(sample.shape[0], 1)
                obs = torch.cat((obs, batch_id), dim=-1).float()
                next_obs = torch.cat((next_obs, batch_id), dim=-1).float()

            with torch.no_grad():
                    
                q_values = agent.q_network(obs)

                considered_q_values = q_values + (action_mask-1.0)*9999.0
                best_actions = torch.argmax(considered_q_values, dim=1)#.reshape(1)

                actions = actions.squeeze()

                assert torch.all(torch.sum(action_mask, 1) >0)
                mask = (actions == best_actions).float()
                proba_rd = epsilon/torch.sum(action_mask, 1)
                probability = mask*(1.0-epsilon+proba_rd) + (1.0-mask)*proba_rd
                
                current_likelyhood.append(probability.cpu().numpy())
                past_likelyhood.append(old_probability.squeeze().cpu().numpy()) 

        others_current_likelyhood = torch.tensor(np.prod([current_likelyhood[:n]+current_likelyhood[n+1:] for n in range(len(agents))], axis=1)).cpu()
        others_past_likelyhood = torch.tensor(np.prod([past_likelyhood[:n]+past_likelyhood[n+1:] for n in range(len(agents))], axis=1)).cpu()
        return others_current_likelyhood.T, others_past_likelyhood.T


def add_ratios(sample, agents, epsilon, single_agent, completed_episodes=None, writer=None, use_state=False):
    
    n_agents = len(agents)
    if use_state:
        sample = sample[:,0]
        repeated_state = sample['observations'].unsqueeze(1).repeat(1, n_agents, 1)
        sample['observations'] = repeated_state
        repeated_next_state = sample['next_observations'].unsqueeze(1).repeat(1, n_agents, 1)
        sample['next_observations'] = repeated_next_state
        sample['index'] = sample['index'].unsqueeze(1).repeat(1, n_agents)
        if '_weight' in sample.keys():
            sample['_weight'] = sample['_weight'].unsqueeze(1).repeat(1, n_agents)

    sample = TensorDict(sample, batch_size=[sample.shape[0], n_agents])

    td_error = torch.stack([agent.get_td_error(sample[:,id_agent]).to('cpu') for id_agent, agent in enumerate(agents)], dim=1)
    current_likelyhood, past_likelyhood = current_and_past_others_actions_likelyhood(sample, agents, epsilon, single_agent)

    sample.set("td_error",td_error)
    sample.set("td-past",td_error/past_likelyhood)
    sample.set("td-cur-past",td_error*current_likelyhood/past_likelyhood)
    sample.set("td-cur",td_error*current_likelyhood)
    sample.set("cur-past",current_likelyhood/past_likelyhood)
    sample.set("cur",current_likelyhood)
    sample.set("past",past_likelyhood)


    if (completed_episodes is not None) and (writer is not None):
        for key in ['td_error', 'td-past', 'td-cur-past', 'td-cur', 'cur-past', 'cur']:

            writer.add_scalar('Mean ratios/'+ key, sample[key].mean(), global_step=completed_episodes)
            writer.add_scalar('Std ratios/'+ key, sample[key].std(), global_step=completed_episodes)

        writer.add_scalar('Mean ratios/past', past_likelyhood.mean(), global_step=completed_episodes)
        writer.add_scalar('Std ratios/past', past_likelyhood.std(), global_step=completed_episodes)

    return sample

def make_env(scenario_name, benchmark=False):
    '''
    Only for MPE
    '''
    from mpe.environment import MultiAgentEnv
    import mpe.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:        
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def create_env(env_id, params, render_mode=None):
    if env_id == 'simultaneous':
        from myenvs.simultaneous_attack import SimultaneousEnv
        env = SimultaneousEnv(n_agents=params['n_agents'], n_actions=params['n_actions'])#, common_reward=params['enforce_coop'])
    elif env_id == 'water-bomber':
        from myenvs.water_bomber import WaterBomberEnv
        env = WaterBomberEnv(x_max=params['x_max'], y_max=params['y_max'], t_max=params['t_max'], n_agents=params['n_agents'], obs_normalization=params['env_normalization'], deterministic=params['deterministic_env'], add_id=params['add_id'])
    elif env_id == 'smac':
        assert params['map'] in ['10m_vs_11m', '27m_vs_30m', '2c_vs_64zg', '2s3z', '2s_vs_1sc', '3s5z', '3s5z_vs_3s6z', '3s_vs_5z', 'bane_vs_bane', 'corridor', 'MMM', 'MMM2']
        import smaclite  
        env = gym.make(f"smaclite/"+params['map'])
    elif env_id == 'lbf':
        import lbforaging
        n_agents = str(params['n_agents'])
        coop = 'coop-' if params['enforce_coop'] else ''
        env_name = "Foraging-5x5-"+n_agents+"p-"+n_agents+"f-"+coop+"v2"
        print("env_name:", env_name)
        env = gym.make(env_name)
    elif env_id == 'mpe':
        assert params['map'] in ['simple_adversary', 'simple_crypto', 'simple_push', 'simple_reference', 'simple_speaker_listener', 'simple_spread', 'simple_tag', 'simple_world_comm', 'simple']
        return make_env(params['map'])
    else:
        raise NameError('Unknown env:'+env_id)
    
    if params['use_state']:
        env = StateWrapper(env)
    return env

def get_n_likeliest(batch, key, n):
    descending = batch[key].sort(dim=0, descending=True)
    descending = batch[key] >= descending.values[n]
    descending = descending.reshape(-1)
    descending &= torch.cumsum(descending, 0) <= n
    return batch[descending]

def get_agents_distrib(batch, q_agents, epsilon):
    concat_distrib = []
    for agent_id, agent in enumerate(q_agents):
        obs = batch[:,agent_id]["observations"]
        avail_actions = batch[:,agent_id]["action_mask"]
        distrib = agent.get_distrib(obs, avail_actions, epsilon)
        concat_distrib.append(distrib)

    concat_distrib = torch.stack(concat_distrib, dim=1)
    return concat_distrib
class StateWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env.reset()
        state = self.env.get_state()
        high = np.ones(np.size(state))
        self.observation_space = Box(-high, high)

    def observation(self, obs):
        return self.env.get_state()

class QAgent():
    def __init__(self, env, agent_id, params, obs_shape, act_shape, writer, experiment_hash=None):
        for k, v in params.items():
            if k=='none':
                setattr(self, None, v)
            else:
                setattr(self, k, v)

        self.params = params
        self.writer = writer
        self.experiment_hash = experiment_hash
        self.env = env
        self.agent_id  = agent_id
        self.one_hot_id = torch.eye(params['n_agents'])[agent_id]

        try:
            self.action_space = env.action_space[agent_id]
        except:
            self.action_space = env.action_space(agent_id)

        if self.dueling:
            network_class=DuelingQNetwork
        else:
            network_class=QNetwork

        if self.params['add_id']:  
            obs_shape += len(self.one_hot_id)
        self.q_network = network_class(obs_shape, act_shape).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.target_network = network_class(obs_shape, act_shape).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_distrib(self, obs, avail_actions, epsilon):
        obs = obs.to(self.device)
        avail_actions = avail_actions.cpu()
        with torch.no_grad():
            obs = torch.Tensor(obs).float()
            
            q_values = self.q_network(obs).cpu()
            considered_q_values = q_values + (avail_actions-1.0)*99999.0
            action = torch.argmax(considered_q_values, dim=1)
  
        avail_actions = torch.tensor(avail_actions)
        avail_actions_ind = np.nonzero(avail_actions)
        proba_select_explo = epsilon/torch.sum(avail_actions, dim=1)
        probability = avail_actions*proba_select_explo.reshape((-1, 1))
        probability = torch.nan_to_num(probability, nan=0.0)
        probability[:,action] += 1.0 - epsilon

        return probability.cpu()

    def act(self, obs, avail_actions, others_explo=None, training=True):
        """
        Greedy action
        """
        if self.params['add_id']:   
            obs = torch.cat((obs, self.one_hot_id), dim=-1).float()
        
        with torch.no_grad():
            obs = torch.Tensor(obs).float()
            q_values = self.q_network(obs.to(acting_device)).cpu()

            assert sum(avail_actions)>0, avail_actions
            considered_q_values = q_values + (avail_actions-1.0)*1e10
            action = torch.argmax(considered_q_values).reshape(1)

        avail_actions = torch.tensor(avail_actions)
        avail_actions_ind = np.nonzero(avail_actions).reshape(-1)

        if int(action) not in avail_actions_ind:
            #print("Unavailable action was choosen!")
            action = torch.tensor([np.random.choice(avail_actions_ind)])

        return float(action)
    
    def train(self, sample, completed_episodes):
        sample = sample.to(self.device)

        obs = sample['observations'].float()
        action_mask = sample['action_mask']
        next_obs = sample['next_observations'].float()
        next_action_mask = sample['next_action_mask']
        reward = sample['rewards']
        dones = sample['dones']
        actions = sample['actions']
        weights = sample['weights']

        if self.params['add_id']: 
            batch_id = self.one_hot_id.repeat(self.params['batch_size'], 1)
            obs = torch.cat((obs, batch_id), dim=-1).float()
            next_obs = torch.cat((next_obs, batch_id), dim=-1).float()

        assert torch.all(torch.sum(action_mask, 1) >0), (obs,action_mask)
        assert torch.all(torch.sum(next_action_mask, 1) >0), action_mask
        
        with torch.no_grad():
            target_max, _ = (self.target_network(next_obs)*next_action_mask).max(dim=1)
            td_target = reward.flatten() + self.gamma * target_max * (1 - dones.flatten())
        old_val = (self.q_network(obs)*action_mask).gather(1, actions).squeeze()

        #weights = torch.ones(self.batch_size).to(self.device)
        
        if self.loss_correction_for_others not in [None, 'none']:
            assert self.loss_correction_for_others in sample.keys()

            others_correction = sample[self.loss_correction_for_others]
            
            if 'sqrt' in self.correction_modification:
                others_correction = torch.sqrt(others_correction)
            if 'sigmoid' in self.correction_modification:
                others_correction = sigmoid(others_correction)
            if 'normalize' in self.correction_modification:
                others_correction /= torch.max(others_correction)
            
            if self.clip_correction_after is not None:
                others_correction = torch.clip(others_correction, -self.clip_correction_after, self.clip_correction_after)
            weights *= others_correction

        td_target = td_target.to(self.device)
        old_val = old_val.to(self.device)
        weights = weights.to(self.device)

        loss = weighted_mse_loss(td_target, old_val, weights)

        with torch.no_grad():
            td_error = torch.abs(td_target-old_val)
        self.writer.add_scalar(str(self.agent_id)+"/td_loss", loss, completed_episodes)
        self.writer.add_scalar(str(self.agent_id)+"/q_values", old_val.mean().item(), completed_episodes)

        self.writer.flush()

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return td_error.cpu().detach().numpy()

    def save(self, path):
        model_path = path / f"{self.agent_id}.iql_model"
        
        torch.save(self.q_network.state_dict(), model_path)

    def load(self, model_path):
        print("model_path", model_path)
        self.q_network.load_state_dict(torch.load(model_path))
        print(f"model sucessfullt loaded from to {model_path}")
        
    def __str__(self):
        pprint(self.__dict__)
        return ""


    def get_td_error(self, sample):

        sample = sample.to(self.device)
        obs = sample['observations'].float()
        action_mask = sample['action_mask']
        next_obs = sample['next_observations'].float()
        next_action_mask = sample['next_action_mask']
        reward = sample['rewards']
        dones = sample['dones']
        actions = sample['actions']

        if self.params['add_id']: 
            batch_id = self.one_hot_id.repeat(sample.shape[0], 1).to(self.device) #10*self.params['batch_size']
            obs = torch.cat((obs, batch_id), dim=-1).float()
            next_obs = torch.cat((next_obs, batch_id), dim=-1).float()
        
        with torch.no_grad():
            target_max, _ = (self.target_network(next_obs)*next_action_mask).max(dim=1)
            td_target = reward.flatten() + self.gamma * target_max * (1 - dones.flatten())
            old_val = (self.q_network(obs)*action_mask).gather(1, actions).squeeze()

            td_error = torch.abs(td_target-old_val)
        return td_error
 
    def importance_weight(self, sample, completed_episodes):
        num, denom = self.current_and_past_others_actions_likelyhood(sample, completed_episodes)
        return (num/denom).to(self.device)
    
def visualize_trajectory(env, agents, completed_episodes):
    arrows = {1:(1,0), 3:(-1,0), 2:(0,1), 0:(0,-1)}

    n_obs, info = env.reset(return_info=True, deterministic=True)
    n_action_mask = info['avail_actions']
    states = [env.get_state()[:-1]]

    q_values = np.zeros((env.X_MAX+1, env.Y_MAX+1))
    choosen_act = np.full((env.X_MAX+1, env.Y_MAX+1), 4,  dtype=int)
    terminated = False
    while not terminated:
        actions = []
        for agent_id, obs in enumerate(n_obs):

            x, y = env.water_bombers[agent_id]
            action_mask = n_action_mask[agent_id]
            
            other_act_randomly = torch.zeros(env.n_agents - 1)
            
            obs = torch.tensor(obs).to(agents[agent_id].device)
            pred = agents[agent_id].q_network(obs).detach().cpu()
            target = pred + (action_mask-1)*9999.0
            target_max = target.max().float()

            target_argmax = target.argmax()

            q_values[ x, y] = target_max
            choosen_act[ x, y] = target_argmax
            actions.append(target_argmax)

        n_previous_action_mask = n_action_mask
        n_next_obs, n_reward, n_terminated, info = env.step(actions)
        
        n_action_mask = info['avail_actions']

        n_obs = n_next_obs
        state = env.get_state()[:-1]
        terminated = n_terminated[0] or (state in states)
        states.append(state)

    fig, ax = plt.subplots()
    im = ax.imshow(q_values.T[::-1])

    fig.colorbar(im, ax=ax, label='Interactive colorbar')

    for x in range(env.X_MAX+1):
        for y in range(env.Y_MAX+1):
            if choosen_act[x,y] != 4:
                plt.arrow(x, env.Y_MAX-y, scale*arrows[choosen_act[x,y]][0], scale*arrows[choosen_act[x,y]][1], head_width=0.1) #, color=

    agents[0].writer.add_figure("q_values_imgs", fig, completed_episodes)


def training_step(params, replay_buffer, smaller_buffer, q_agents, completed_episodes, training, writer):
    for q_agent in q_agents:
        q_agent.q_network = q_agent.q_network.to(q_agent.device)
        q_agent.target_network = q_agent.target_network.to(q_agent.device)

    epsilon = linear_schedule(params['start_e'], params['end_e'], params['exploration_fraction'] * params['total_timesteps'], completed_episodes)
    n_agents = len(q_agents)
    
    if training and completed_episodes > params['learning_starts'] and completed_episodes % params['train_frequency'] == 0:
        maybe_writer = writer if completed_episodes % params['evaluation_frequency'] == 0 else None
        if params['rb'] =='laber':
            # On met a jour les TD errors 
            big_sample = replay_buffer.sample()
            big_index = big_sample['index']
            big_sample = add_ratios(big_sample, q_agents, epsilon, params['single_agent'], use_state=params['use_state'], completed_episodes=completed_episodes, writer=maybe_writer)
            smaller_buffer.extend(big_sample)
            sample = smaller_buffer.sample()
            index = big_index[sample['index']][:,0]
            #writer.add_histogram('distribution centers', index.reshape(-1), completed_episodes, bins=10)

            values = np.array(index).astype(float).reshape(-1)
            counts, limits= np.histogram(values, bins=10, range=(0.0, params['buffer_size']))

            sum_sq = values.dot(values)
            writer.add_histogram_raw(
                tag='distribution centers',
                min=values.min(),
                max=values.max(),
                num=len(values),
                sum=values.sum(),
                sum_squares=sum_sq,
                bucket_limits=limits[1:].tolist(),
                bucket_counts=counts.tolist(),
                global_step=completed_episodes)
            writer.flush()
            if params['prioritize_big_buffer']:
                replay_buffer.update_tensordict_priority(big_sample)

        elif params['rb'] =='likely':
            big_sample = replay_buffer.sample()
            big_sample = add_ratios(big_sample, q_agents, epsilon, params['single_agent'], use_state=params['use_state'], completed_episodes=completed_episodes, writer=maybe_writer)
            sample = torch.stack([get_n_likeliest(big_sample[:,agent_id], params['filter'], params['batch_size']) for agent_id in range(n_agents)], dim=1)
            writer.add_histogram('distribution centers', sample['index'].reshape(-1), completed_episodes)
            if params['prioritize_big_buffer']:
                replay_buffer.update_tensordict_priority(big_sample)
        
        else:
            sample = replay_buffer.sample()
            if not (params['env_id'] == 'mpe' and params['map'] == 'simple'):
                sample = add_ratios(sample, q_agents, epsilon, params['single_agent'], use_state=params['use_state'], completed_episodes=completed_episodes, writer=maybe_writer)
        
        #samples = [sample for _ in range(n_agents)]
        #writer.add_histogram('distribution centers', index.reshape(-1), completed_episodes)
        weights = torch.ones((len(sample),n_agents))

        if '_weight' in sample.keys() and params['correct_prio']:
            weights = sample['_weight']
        
        weights = weights/weights.sum()
        #weights.sum()/weights
        #weights /= weights.max()

        sample['weights'] = weights #.repeat((n_agents, 1)).T

        td_errors = []
        for agent_id, agent in enumerate(q_agents):
            
            agent_td_error = agent.train(sample[:,agent_id], completed_episodes)
            td_errors.append(agent_td_error)

        if params['rb'] == 'prioritized':
            sample.set("td_error", torch.tensor(td_errors).T)
            replay_buffer.update_tensordict_priority(sample)    

    # update target network
    for agent in q_agents:
        if completed_episodes % agent.target_network_frequency == 0:
            for target_network_param, q_network_param in zip(agent.target_network.parameters(), agent.q_network.parameters()):
                target_network_param.data.copy_(
                    agent.tau * q_network_param.data + (1.0 - agent.tau) * target_network_param.data
                )

    return q_agents

def run_episode(env, q_agents, completed_episodes, params, replay_buffer=None, smaller_buffer=None, training=False, visualisation=False, verbose=False, plot_q_values=False, writer=None):
    for q_agent in q_agents:
        q_agent.q_network = q_agent.q_network.to('cpu')
        q_agent.target_network = q_agent.target_network.to('cpu')
    
    if training:
        assert replay_buffer is not None
        if params['rb'] == 'laber':
            assert smaller_buffer is not None
    
    if plot_q_values:
        assert params['env_id'] == 'water_bomber'
        visualize_trajectory(env, q_agents, completed_episodes)

    n_obs, info = env.reset(return_info=True)
    n_action_mask = info['avail_actions']
    n_agents = len(q_agents)
    terminated = False
    episode_reward = np.zeros((n_agents,))
    nb_steps = 0
    n_next_obs = None
    n_previous_action_mask = None
    epsilon = linear_schedule(params['start_e'], params['end_e'], params['exploration_fraction'] * params['total_timesteps'], completed_episodes)
    #if not training:
    #    epsilon = 0.0

    n_obs = torch.tensor(n_obs)

    n_act_randomly = [params['random_policy'] or (random.random() < epsilon and training) for _ in range(env.n_agents)]
    n_other_act_randomly = torch.tensor([n_act_randomly[:agent_id]+n_act_randomly[agent_id+1:] for agent_id in range(n_agents)])

    if params['add_epsilon']:
        if params['use_state']:
            n_epsilon = torch.full((1,1), epsilon)
        else:
            n_epsilon = torch.full((n_agents,1), epsilon)

        if len(n_obs.shape)==1:
            n_epsilon = n_epsilon.reshape(-1)
        
        n_obs = torch.cat((n_obs, n_epsilon), dim=-1).float()
    if params['add_others_explo']:
        if params['use_state']:
            n_other_act_randomly = torch.tensor(n_act_randomly).reshape(-1)

        if len(n_obs.shape)==1:
            n_other_act_randomly = torch.tensor(n_other_act_randomly).reshape(-1)
        n_obs = torch.cat((n_obs, n_other_act_randomly.float()), dim=-1).float()
    
    while not terminated:
        if visualisation:
            sleep(0.1)
            env.render()  # Uncomment for rendering

        n_action = []
        n_probabilities = []
            
        enum_obs = [(i, n_obs) for i in range(n_agents)] if params['use_state'] else enumerate(n_obs)
        for agent_id, obs in enum_obs:
            avail_actions = n_action_mask[agent_id]
            avail_actions_ind = np.nonzero(avail_actions)[0]
            if n_act_randomly[agent_id]:
                action = np.random.choice(avail_actions_ind).reshape(-1)
                probability = epsilon/sum(avail_actions)
            else:
                action = q_agents[agent_id].act(obs, avail_actions, others_explo=n_other_act_randomly[agent_id])
                probability = 1.0-epsilon*(sum(avail_actions)-1)/sum(avail_actions) #if training else 1.0

            assert probability > 0.0 , (probability, epsilon)
            
            
            n_action.append(action)
            n_probabilities.append(probability)

        if False:
            print("n_previous_action_mask", n_previous_action_mask)
            print("n_action", n_action)
            #env.render()

        n_previous_action_mask = n_action_mask
        n_action = [int(a) for a in n_action]
        n_next_obs, n_reward, n_terminated, n_truncated, info = env.step(n_action)
        n_terminated = n_terminated or n_truncated

        n_act_randomly = [params['random_policy'] or (random.random() < epsilon and training) for _ in range(env.n_agents)]
        n_other_act_randomly = torch.tensor([n_act_randomly[:agent_id]+n_act_randomly[agent_id+1:] for agent_id in range(n_agents)])

        n_next_obs = torch.tensor(n_next_obs)

        if params['add_epsilon']:
            if params['use_state']:
                n_epsilon = torch.full((1,1), epsilon)
            else:
                n_epsilon = torch.full((n_agents,1), epsilon)
            if len(n_obs.shape)==1:
                n_epsilon = n_epsilon.reshape(-1)
            
            n_next_obs = torch.cat((n_next_obs, n_epsilon), dim=-1).float()
        if params['add_others_explo']:
            if params['use_state']:
                n_other_act_randomly = torch.tensor(n_act_randomly).reshape(-1)
            if len(n_obs.shape)==1:
                n_other_act_randomly = n_other_act_randomly.reshape(-1)
            n_next_obs = torch.cat((n_next_obs, n_other_act_randomly.float()), dim=-1).float()

        if np.array(n_reward).size ==1:
            n_reward = np.full((n_agents, 1), n_reward)
        if np.array(n_terminated).size ==1:
            n_terminated = np.full((n_agents, 1), n_terminated)

        n_next_obs = torch.tensor(n_next_obs, dtype=torch.float)
        n_action_mask = info['avail_actions']

        if False:
            print("n_next_obs", n_next_obs)
            print("n_reward", n_reward)
            print("n_terminated", n_terminated)
            print("n_action_mask", n_action_mask)
            print()

        episode_reward += np.array(n_reward).squeeze()

        #if training: On ajoute au rb meme quand on explore pas

        if replay_buffer is not None and not params['fixed_buffer']:    
            for previous_action_mask, action_mask in zip(n_previous_action_mask, n_action_mask):
                assert sum(previous_action_mask) > 0 
                assert sum(action_mask) > 0 

            if params['use_state']:
                observations = torch.tensor(n_obs, dtype=torch.float).reshape(1, -1)
                next_observations = torch.tensor(n_next_obs, dtype=torch.float).reshape(1, -1)
            else:
                observations = torch.tensor(n_obs, dtype=torch.int64).reshape(n_agents, -1)
                next_observations = torch.tensor(n_next_obs, dtype=torch.int64).reshape(n_agents, -1)
            
            transition = {
                'action_mask':torch.tensor(n_previous_action_mask, dtype=torch.int64).reshape(n_agents, -1),
                'actions':torch.tensor(n_action, dtype=torch.int64).reshape(n_agents, -1),
                'actions_likelihood':torch.tensor(n_probabilities, dtype=torch.float).reshape(n_agents, -1),
                'rewards':torch.tensor(n_reward, dtype=torch.float).reshape(n_agents, -1),
                'next_action_mask':torch.tensor(n_action_mask, dtype=torch.int64).reshape(n_agents, -1),
                'dones':torch.tensor(n_terminated, dtype=torch.float).reshape(n_agents, -1),
            }
            transition = TensorDict(transition, batch_size=n_agents)
            if params['use_state']:
                transition = TensorDict(transition.unsqueeze(0), batch_size=[1])
            transition['observations'] = observations
            transition['next_observations'] = next_observations
            replay_buffer.add(transition)

        nb_steps += 1
        n_obs = n_next_obs
        terminated = n_terminated[0] or nb_steps > params['t_max']

    return nb_steps, episode_reward 
    
def run_training(env_id, verbose=True, run_name='', path=None, **args):
    with open(Path('src/config/'+env_id+'/default.yaml')) as f:
        params = yaml.safe_load(f)

    
    if path is None:
        path = Path.cwd() / 'results'
    
    for k, v in args.items():
        if v is not None:
            params[k] = v
    #if __name__ == "__main__":
    pprint(params)
    old_params = copy(params)

    if params['device'] == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif params['device'] == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    env = create_env(env_id, params)
    visu_env = create_env(env_id, params)

    print("writer path: ", path/run_name)
    writer = SummaryWriter(path/run_name) #
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in params.items()])),
    )
    #writer.add_hparams(vars(args), {})
    writer.flush()


    # TRY NOT TO MODIFY: seeding
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    torch.backends.cudnn.deterministic = params['torch_deterministic']

    ### Creating Env
    env.reset()
    if params['use_state']:
        obs_shape = env.observation_space.shape
    else:
        try:
            obs_shape = env.observation_space[-1].shape
        except:
            #obs_shape = env.observation_space(0).shape
            obs_shape = list(env.observation_space(0).values())[0].shape
        
    size_obs = int(np.product(obs_shape))
    
    try:
        size_act = int(env.action_space[-1].n)
    except:
        #size_act = int(env.action_space(0).n)
        size_act = int(env.action_space(0).n)
    
    ## Increasing the state size for state augmentation
    if params['add_epsilon']:
        size_obs += 1
    if params['add_others_explo']:
        size_obs += env.n_agents - 1
        if params['use_state']:
            size_obs += 1


    if verbose:
        print('-'*20)
        print('num_agents: ',env.n_agents)
        print('size_obs: ',size_obs)    
        print('size_act: ',size_act)    
        print('-'*20)

    
    ### Creating replay buffer
    if params['load_buffer_from'] is not None:
        rb_path = Path(params['load_buffer_from'])
    else:
        rb_path = path

    def add_to_rb(rb, path, id_max=None):
        data = torch.load(list_paths[-1])
        non_empty_mask = data['observations'].abs().sum(dim=2).bool().reshape(-1)
        
        data = data[non_empty_mask]
        if id_max is not None:
            data = data[:id_max]
        replay_buffer.extend(data)
    
    replay_buffer, smaller_buffer = create_rb(rb_type=params['rb'], buffer_size=params['buffer_size'], batch_size=params['batch_size'], n_agents=env.n_agents, device='cpu', prio=params['prio'], prioritize_big_buffer=params['prioritize_big_buffer'], path=path/run_name/'replay_buffer' if params['buffer_on_disk'] else None) #params['device']
    if params['load_buffer_from'] is not None:
        #replay_buffer, smaller_buffer = create_rb(rb_type=params['rb'], buffer_size=params['buffer_size'], batch_size=params['batch_size'], n_agents=env.n_agents, device=params['device'], prio=params['prio'], prioritize_big_buffer=params['prioritize_big_buffer'], path=path/run_name/'replay_buffer' if params['buffer_on_disk'] else None)
        rb_path = str(Path(params['load_buffer_from'])/'rb') #/ 'replay_buffer.pt')
        """bs = replay_buffer._batch_size
        rb_path = str(Path(params['load_buffer_from']) / 'rb' / 'final')
        
        print("loading buffer from", rb_path)
        snapshot = torchsnapshot.Snapshot(path=rb_path)
        target_state = {
            "state": replay_buffer
        }
        snapshot.restore(app_state=target_state)
        print("Replay buffer saved to", rb_path)
        replay_buffer._batch_size = bs"""

        list_paths = [ f.path for f in os.scandir(rb_path) if f.is_file() ]
        list_paths.sort()

        if params['last_buffer']:
            add_to_rb(replay_buffer, list_paths[-1])
        else:
            frac = int(params['buffer_size'] / len(list_paths))+1

            for sub_rb_path in list_paths:
                print("sub_rb_path", sub_rb_path)
                add_to_rb(replay_buffer, sub_rb_path, frac)
        #with open(rb_path, 'rb') as handle:
        #    data = pickle.load(handle)


    ### Creating Agents
    q_agents = [QAgent(env, a, params, size_obs, size_act, writer, run_name)  for a in range(env.n_agents)]
    
    if params['load_agents_from'] is not None:
        for name, agent in enumerate(q_agents):
            model_path = path / f"{params['load_agents_from']}/saved_models/{name}.iql_model"
            agent.load(model_path)

    if params['single_agent']:
        agent_0 = q_agents[0]
        for agent in range(env.n_agents):
            q_agents[agent].q_network = agent_0.q_network
            
            assert q_agents[agent].q_network is agent_0.q_network
        
    results = []
    pbar=trange(params['total_timesteps'])
    for completed_episodes in pbar:
        # Training episode
        if not params['visualisation']:
            q_agents = training_step(params, replay_buffer, smaller_buffer, q_agents, completed_episodes, True, writer)
            if not params['fixed_buffer']:
                run_episode(env, q_agents, completed_episodes, params, replay_buffer=replay_buffer, smaller_buffer=smaller_buffer, training=True, visualisation=False, verbose=False, writer=writer)
                

        # Evaluation episode
        if completed_episodes % params['evaluation_frequency'] == 0:
            if params['plot_q_values']:
                run_episode(env, q_agents, completed_episodes, params, replay_buffer=replay_buffer, smaller_buffer=smaller_buffer, training=False, plot_q_values=True, writer=writer)
            
            #list_total_reward = []
            agents_total_rewards = []
            #[[] for a in range(n_agents)]
            average_duration = 0.0

            for eval in range(params['evaluation_episodes']):
                
                if params['visualisation'] and eval==0:
                    nb_steps, total_reward = run_episode(visu_env, q_agents, completed_episodes, params, replay_buffer=replay_buffer, smaller_buffer=smaller_buffer, training=False, visualisation=True)
                else:
                    nb_steps, total_reward = run_episode(env, q_agents, completed_episodes, params, replay_buffer=replay_buffer, smaller_buffer=smaller_buffer, training=False, visualisation=False)

                #for a in range(n_agents):
                agents_total_rewards.append(total_reward)
                #episode_reward += np.mean(n_reward)
                #average_reward 
                #list_total_reward.append(list(n_reward))
                average_duration += nb_steps
            
            average_duration /= params['evaluation_episodes']
            agents_total_rewards = np.array(agents_total_rewards)
            agents_total_rewards = np.mean(agents_total_rewards, axis=0).squeeze()
            if params['env_id'] == 'mpe' and params['map'] == 'simple':
                agents_total_rewards = [agents_total_rewards]

            for a in range(len(q_agents)):
                writer.add_scalar(str(a)+"/Average Return", agents_total_rewards[a], completed_episodes)
                
            if params['env_id'] == 'lbf':
                average_return = np.sum(agents_total_rewards)
            else:
                average_return = np.mean(agents_total_rewards)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("Average Return", average_return, completed_episodes)
            writer.add_scalar("Size replay buffer", len(replay_buffer), completed_episodes)
            
            if params['track']:
                wandb.log({
                        "Average Return":average_return, 
                        "Completed Episodes:":completed_episodes
                    })
            pbar.set_description(f"Return={average_return:5.1f}") #, Duration={average_duration:5.1f}"
            results.append(average_return)



        stored_transitions = completed_episodes*params['t_max']    
        if params['save_buffer'] and (stored_transitions % params['buffer_size'])==0 and (stored_transitions // params['buffer_size'])!=0:
            k = (stored_transitions // params['buffer_size'])-1
            os.makedirs(path/ run_name / 'rb', exist_ok=True)
            rb_path = str(path/ run_name / 'rb' / ('replay_buffer_'+str(k)+'.pt'))
            #"""
            #rb_path = path/ run_name / 'replay_buffer.pickle'
            torch.save(replay_buffer[:len(replay_buffer)], rb_path)
            print("Replay buffer saved to", rb_path)

            #"""
            

            #state = {"state": replay_buffer}
            #snapshot = torchsnapshot.Snapshot.take(app_state=state, path=rb_path)
                    
                
    env.close() 
    visu_env.close()

    # Savings
    if params['save_buffer']:
        stored_transitions = completed_episodes*params['t_max']    
        #rb_path = str(path/ run_name / 'rb' / 'final')
        #os.makedirs(rb_path, exist_ok=True)
        #"""
        os.makedirs(path/ run_name / 'rb', exist_ok=True)
        k = (stored_transitions // params['buffer_size'])
        rb_path = str(path/ run_name / 'rb' / ('replay_buffer_'+str(k)+'.pt'))
        torch.save(replay_buffer[:len(replay_buffer)], rb_path)
        #with open(rb_path, 'wb') as handle:
        #"""
        

        #state = {"state": replay_buffer}
        #snapshot = torchsnapshot.Snapshot.take(app_state=state, path=rb_path)
        print("Replay buffer saved to", rb_path)
            
    steps = [i for i in range(0, params['total_timesteps'], params['evaluation_frequency'])]    
    results_dict = {
            'Average optimality': results,
            'Step': steps,
        }
    result_df = pd.DataFrame(results_dict)

    os.makedirs(path / run_name, exist_ok=True)
    if params['save_model']:
        model_path = path/ run_name / 'saved_models'
        os.makedirs(model_path, exist_ok=True)
        for agent in q_agents:
            agent.save(model_path)

    with open(path/ run_name/'params.yaml', 'w') as f:
        yaml.dump(params, f, default_flow_style=False)

    result_df = result_df.assign(**params)
    result_df.to_csv(path / run_name / 'results.csv', index=False)

    assert params == old_params, (params, old_params)
    return steps, results


def main(**params):

    params["run_name"] = params["run_name"] if params["run_name"] is not None else ''
    params["run_name"] += '_{date:%Y-%m-%d_%H:%M:%S}'.format( date=datetime.datetime.now() ) 

    print("Run name:", params["run_name"])
        
    steps, results = run_training(**params)

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))