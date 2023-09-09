#from water_bomber_gym import WaterBomberEnv
#from simultaneous_env import SimultaneousEnv
from myenvs.simultaneous_env import *
from myenvs.water_bomber_gym import *
from torch.distributions.categorical import Categorical

import random
import yaml
import numpy as np
from time import sleep
from copy import deepcopy
import pickle
import pandas as pd 
import sys
scale = 0.25   
import wandb
from dict_hash import dict_hash
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import os
from pathlib import Path
import random
import time
from distutils.util import strtobool
from tqdm import trange
from pprint import pprint
from copy import copy
#import gymnasium as gym
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import smaclite  # noqa

from collections import Counter
#from stable_baselines3 import DQN

#from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
#from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl

#from smac.env.pettingzoo import StarCraft2PZEnv
from torch.utils.tensorboard import SummaryWriter

import contextlib

from pettingzoo.test import api_test

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("error", category=RuntimeWarning)


import matplotlib.pyplot as plt

#from gymnasium.spaces import *
from gym.spaces import *

from tensordict import tensorclass, TensorDict
from torchrl.data.replay_buffers.samplers import RandomSampler, PrioritizedSampler
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage, LazyMemmapStorage, ListStorage, TensorDictPrioritizedReplayBuffer

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    #parser.add_argument("--env-id", default='Water-bomber-v1',
    parser.add_argument("--env-id", choices=['simultaneous', 'water-bomber', 'smac'] ,default='simultaneous',
        help="the id of the environment")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--device", type=str, choices=['cpu', 'mps', 'cuda'], nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")
    parser.add_argument("--use-state", type=lambda x: bool(strtobool(x)), nargs="?", const=True,
        help="whether we give the global state to agents instead of their respective observation")
    parser.add_argument("--save-buffer", type=lambda x: bool(strtobool(x)), nargs="?", const=True)
    parser.add_argument("--load-buffer", type=lambda x: bool(strtobool(x)), nargs="?", const=True)
    parser.add_argument("--save-imgs", type=lambda x: bool(strtobool(x)), nargs="?", const=True,
        help="whether to save images of the V or Q* functions")
    parser.add_argument("--run-name", type=str, default=None)
    

    # Environment specific arguments
    parser.add_argument("--x-max", type=int)
    parser.add_argument("--y-max", type=int)
    parser.add_argument("--t-max", type=int)
    parser.add_argument("--n-agents", type=int)
    parser.add_argument("--env-normalization", type=lambda x: bool(strtobool(x)), nargs="?", const=True)
    parser.add_argument("--num-envs", type=int, 
        help="the number of parallel game environments")

    # Algorithm specific arguments
    parser.add_argument("--load-agents-from", type=str, default=None,
        help="the experiment from which to load agents.")
    parser.add_argument("--load-buffer-from", type=str, default=None,
        help="the experiment from which to load agents.")
    parser.add_argument("--random-policy", type=lambda x: bool(strtobool(x)), nargs="?", const=True)
    parser.add_argument("--no-training", type=lambda x: bool(strtobool(x)), nargs="?", const=True,
        help="whether to show the video")
    parser.add_argument("--total-timesteps", type=int, #
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, #default=1e-3,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, #
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, #default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, #default=1.,
        help="the target network update rate")
    parser.add_argument("--evaluation-frequency", type=int, #default=1000
                        )
    parser.add_argument("--evaluation-episodes", type=int, #default=100
                        )
    parser.add_argument("--target-network-frequency", type=int, # 
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int,  #2**18, #256, #
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
    parser.add_argument("--prioritize-big-buffer", type=lambda x: bool(strtobool(x)), nargs="?", const=True)
    parser.add_argument("--dueling", type=lambda x: bool(strtobool(x)), nargs="?", const=True,
        help="whether to use a dueling network architecture.")
    parser.add_argument("--deterministic-env", type=lambda x: bool(strtobool(x)), nargs="?", const=True)
    parser.add_argument("--boltzmann-policy", type=lambda x: bool(strtobool(x)), nargs="?", const=True)
    #parser.add_argument("--loss-corrected-for-others", type=lambda x: bool(strtobool(x)), nargs="?", const=True)
    parser.add_argument("--loss-not-corrected-for-priorisation", type=lambda x: bool(strtobool(x)), nargs="?", const=True)
    parser.add_argument("--prio", choices=['none', 'td_error', 'td-past', 'td-cur-past', 'td-cur', 'cur-past', 'cur'], nargs="?", const=True)
    parser.add_argument("--filter", choices=['none', 'td_error', 'td-past', 'td-cur-past', 'td-cur', 'cur-past', 'cur', 'past'], nargs="?", const=True)
    parser.add_argument("--loss-correction-for-others", choices=['none','td_error', 'td-past', 'td-cur-past', 'td-cur', 'cur-past', 'cur'], default=None)
    parser.add_argument("--sqrt-correction", type=lambda x: bool(strtobool(x)), nargs="?", const=True)
    parser.add_argument("--clip-correction-after", type=float, nargs="?", const=True)
    parser.add_argument("--rb", choices=['uniform', 'prioritized', 'laber', 'likely', 'correction'], default='uniform')
    #parser.add_argument("--multi-agents-correction", choices=['add_epsilon', 'add_probabilities', 'predict_probabilities'])
    parser.add_argument("--predict-others-likelyhood", type=lambda x: bool(strtobool(x)), nargs="?", const=True)
    parser.add_argument("--add-others-explo", type=lambda x: bool(strtobool(x)), nargs="?", const=True)
    args = parser.parse_args()
    # fmt: on
    #assert args.num_envs == 1, "vectorized envs are not supported at the moment"

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
    

def get_correclty_sampled_transitions(q_agents, epsilon, batch_size, batch):
    """
    Select batch_size transitions from replay_buffer that actions follow policy pi.
    """
    
    actions = batch["actions"]
    nb = batch.shape[0]
    #if nb > 1:
    #print("nb:", batch.shape[0])

    agents_distrib = get_agents_distrib(batch, q_agents, epsilon)
    batches = []
    for agent_id, agent in enumerate(q_agents):
        ### BAD: ONLY WORK FOR 2 AGENTS
        # We should create a recusrive sampling for multi agents
        #others_distrib = torch.cat((agents_distrib[:,:agent_id],agents_distrib[:,agent_id+1:]), dim=1)
        #others_actions = torch.cat((actions[:,:agent_id],actions[:,agent_id+1:]), dim=1)
        
        others_distrib = agents_distrib[:,1-agent_id]
        others_actions = actions[:,1-agent_id][0].cpu()
        nb_possible_actions = others_distrib.shape[-1]
        nb_occ = torch.bincount(others_actions.reshape(-1), minlength=nb_possible_actions)
        distrib = others_distrib[:,others_actions] / nb_occ[others_actions]
        assert not torch.isnan(distrib).any(), distrib
        index = Categorical(probs=distrib.reshape(-1)).sample(sample_shape=(batch_size,)) 
        
        
        batch_agent = batch[index]
        batches.append(batch_agent)
    return batches
    #torch.bincount(index)
    #return batch[res][index]
  
def collate_n_state_batch(nb_states, nb_transitions, q_agents, epsilon, replay_buffer):
    """
    Concatenate nb_states mini-batch of nb_transitions transitions samples according to the others current policy
    """

    """batch = replay_buffer[:len(replay_buffer)]
    #pprint(batch)
    state_id = random.randint(0, len(replay_buffer))
    state = batch[state_id,0]["observations"]
    res = torch.all(batch[:,0]["observations"] == state, dim=1)
    """
    batch = replay_buffer[:len(replay_buffer)]
    index = torch.randint(high=len(replay_buffer),size=(nb_states,)) 
    states = batch[index]['observations']
    #print("states", states.shape)

    micro_batches = []
    for state in states[:,0]:
        #print("batch", batch)
        #print("state", state)
        
        res = torch.all(batch[:,0]["observations"] == state, dim=1)
        sub_batch = batch[res]
        micro_batch = get_correclty_sampled_transitions(q_agents, epsilon, nb_transitions, sub_batch)
        micro_batch = torch.stack(micro_batch, dim=1)
        #print("micro_batch", micro_batch)
        micro_batches.append(micro_batch)
    batch = torch.stack(micro_batches, dim=0)
    # need to merge the 2 first dim
    #print("final batch", batch)
    return batch
  








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

        #self.device = torch.device("cuda" if torch.cuda.is_available() and params['cuda'] else "cpu")
        

        self.env = env

        self.agent_id  = agent_id
        try:
            self.action_space = env.action_space[agent_id]
        except:
            self.action_space = env.action_space(agent_id)


        if self.dueling:
            network_class=DuelingQNetwork
        else:
            network_class=QNetwork

        self.q_network = network_class(obs_shape, act_shape).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.target_network = network_class(obs_shape, act_shape).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        #observation_space = env.observation_space(agent_id)
        """observation_space = Dict({
            'observation': env.observation_space(self.agent_id),
            'action_mask': env.observation_space(self.agent_id)['action_mask']
        })"""

        """self.replay_buffer = DictReplayBuffer(
            self.buffer_size,
            env.observation_space(self.agent_id), #env.observation_space(self.agent_id)
            env.action_space(self.agent_id),
            self.device,handle_timeout_termination=False,
            )"""
    
            
        
        #if self.params['load_buffer']:
        #    self.load_rb()

    def get_distrib(self, obs, avail_actions, epsilon):
        obs = obs.to(self.device)
        avail_actions = avail_actions.cpu()
        with torch.no_grad():
            obs = torch.Tensor(obs).float()
            
            q_values = self.q_network(obs).cpu()


            considered_q_values = q_values + (avail_actions-1.0)*99999.0
            action = torch.argmax(considered_q_values, dim=1)

        #assert action in avail_actions_ind
        avail_actions = torch.tensor(avail_actions)
        avail_actions_ind = np.nonzero(avail_actions)


        proba_select_explo = epsilon/torch.sum(avail_actions, dim=1)
        #avail_actions = env.get_avail_agent_actions(agent_id)
        probability = avail_actions*proba_select_explo.reshape((-1, 1))
        probability = torch.nan_to_num(probability, nan=0.0)
        probability[:,action] += 1.0 - epsilon

        return probability.cpu()

    def act(self, obs, avail_actions, epsilon=None, others_explo=None, training=True):
        with torch.no_grad():
            obs = torch.Tensor(obs).float()
            
            q_values = self.q_network(obs.to(self.device)).cpu()


            assert sum(avail_actions)>0, avail_actions
            considered_q_values = q_values + (avail_actions-1.0)*99999.0
            action = torch.argmax(considered_q_values).reshape(1)

        #assert action in avail_actions_ind
        avail_actions = torch.tensor(avail_actions)
        avail_actions_ind = np.nonzero(avail_actions).reshape(-1)

        if int(action) not in avail_actions_ind:
            print("action", action)
            print("avail_actions", avail_actions)
            print("avail_actions_ind", avail_actions_ind)
            print("obs", obs)
            print("considered_q_values", considered_q_values)
            self.env.render()
            #assert False
            
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


        #assert next_observations[0][-2] == self.agent_id
        assert torch.all(torch.sum(action_mask, 1) >0), (obs,action_mask)
        assert torch.all(torch.sum(next_action_mask, 1) >0), action_mask
        
        with torch.no_grad():
            target_max, _ = (self.target_network(next_obs)*next_action_mask).max(dim=1)
            td_target = reward.flatten() + self.gamma * target_max * (1 - dones.flatten())
        #old_val = (self.q_network(obs)*action_mask).gather(1, sample['actions'][self.agent_id].unsqueeze(0)).squeeze()
        old_val = (self.q_network(obs)*action_mask).gather(1, actions).squeeze()

        weights = torch.ones(self.batch_size).to(self.device)
        
        if self.loss_not_corrected_for_priorisation:
            sample['_weight'] = torch.ones(self.batch_size).to(self.device)

        if self.loss_correction_for_others not in [None, 'none']:
            if self.loss_correction_for_others not in sample.keys():
                sample = self.add_ratios(sample, completed_episodes)

            others_correction = sample[self.loss_correction_for_others]
            if self.sqrt_correction:
                others_correction = torch.sqrt(others_correction)
            if self.clip_correction_after is not None:
                others_correction = torch.clip(others_correction, -self.clip_correction_after, self.clip_correction_after)
            weights *= others_correction
            #self.importance_weight(sample, completed_episodes)

        td_target = td_target.to(self.device)
        old_val = old_val.to(self.device)
        weights = weights.to(self.device)

        loss = weighted_mse_loss(td_target, old_val, weights)

        #if self.params['predict_others_likelyhood']:
            #obs = torch.cat((obs, torch.tensor([epsilon])), 0)


        #else:
        #    loss = F.mse_loss(td_target, old_val)
        #td_error = None
        #if self.params['rb'] == 'prioritized':
        with torch.no_grad():
            td_error = torch.abs(td_target-old_val)
            #sample.set("td_error",td_error)
            #sample.set("td",td_error)
            #sample['td']
            #self.replay_buffer.update_tensordict_priority(sample)

        self.writer.add_scalar(str(self.agent_id)+"/td_loss", loss, completed_episodes)
        self.writer.add_scalar(str(self.agent_id)+"/q_values", old_val.mean().item(), completed_episodes)

        self.writer.flush()

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.save_model:
            self.save()

        return td_error.cpu().detach().numpy()

    

    def save(self):
        model_path = Path.cwd() / 'runs' / self.experiment_hash / "{self.agent_id}.cleanrl_model"
        os.makedirs("runs/"+self.experiment_hash+"/saved_models", exist_ok=True)
        torch.save(self.q_network.state_dict(), model_path)
        #print(f"model saved to {model_path}")

    def load(self, model_path):
        model_path = Path.cwd() / 'runs' / self.experiment_hash / "{self.agent_id}.cleanrl_model"
        self.q_network.load_state_dict(torch.load(model_path))
        print(f"model sucessfullt loaded from to {model_path}")
        
    def __str__(self):
        pprint(self.__dict__)
        return ""
    
    def save_rb(self):
        #buffer_path = f"runs/{params['run_name']}/saved_models/{self.agent_id}_buffer.pkl"
        #save_to_pkl(buffer_path, self.replay_buffer)
        env_type = "_det" if self.deterministic_env else "_rd"
        add_eps = "_eps" if self.add_epsilon else ""
        add_expl = "_expl" if self.add_others_explo else ""

        path = Path.cwd() / 'rbs' / Path(str(self.agent_id)+env_type+add_eps+add_expl+'_replay_buffer.pickle')
        with open(path, 'wb') as handle:
            pickle.dump(self.replay_buffer[:], handle)
        
        #self.rb_storage[:]

    def load_rb(self):
        #self.replay_buffer = load_from_pkl(buffer_path)
        #assert isinstance(self.replay_buffer, ReplayBuffer), "The replay buffer must inherit from ReplayBuffer class"
        env_type = "_det" if self.deterministic_env else "_rd"
        add_eps = "_eps" if self.add_epsilon else ""
        add_expl = "_expl" if self.add_others_explo else ""

        path = Path.cwd() / 'rbs' / Path(str(self.agent_id)+env_type+add_eps+add_expl+'_replay_buffer.pickle')
        with open(path, 'rb') as handle:
            data = pickle.load(handle)

        self.replay_buffer.extend(data)

    def get_td_error(self, sample):

        sample = sample.to(self.device)
        obs = sample['observations'].float()
        action_mask = sample['action_mask']
        next_obs = sample['next_observations'].float()
        next_action_mask = sample['next_action_mask']
        reward = sample['rewards']
        dones = sample['dones']
        actions = sample['actions']

        
        with torch.no_grad():
            target_max, _ = (self.target_network(next_obs)*next_action_mask).max(dim=1)
            td_target = reward.flatten() + self.gamma * target_max * (1 - dones.flatten())
            old_val = (self.q_network(obs)*action_mask).gather(1, actions).squeeze()

            td_error = torch.abs(td_target-old_val)
        return td_error

    def visualize_q_values(self, env, completed_episodes):
        arrows = {1:(1,0), 3:(-1,0), 2:(0,1), 0:(0,-1)}

        observations, info = env.reset(return_info=True)
        n_action_mask = info['avail_actions']

        obs = observations[self.agent_id] #['observation']
        #observation['observation'][-1] = 5
        #assert observation[-2] == self.agent_id
        q_values = np.zeros((env.X_MAX+1, env.Y_MAX+1))
        if self.dueling:
            v_values = np.zeros((env.X_MAX+1, env.Y_MAX+1))
        choosen_act = np.zeros((env.X_MAX+1, env.Y_MAX+1), dtype=int)
        #q_values = np.zeros((3,env.X_MAX, env.Y_MAX))
        for x in range(env.X_MAX+1):
            for y in range(env.Y_MAX+1):

                action_mask= env.get_action_mask(x,y)
                obs = torch.tensor(obs).to(self.device)
                pred = self.q_network(obs).detach().cpu()
                target = pred + (action_mask-1)*9999.0
                target_max = target.max().float()

                target_argmax = target.argmax()

                if self.dueling:
                    v_values[ x, y] = self.q_network(obs, value_only=True).detach().cpu()

                #clipped_target_max = (np.clip(target_max, -10, 10) + 10)/ 20
                #q_values[0, x, y] = clipped_target_max 
                #q_values[1, x, y] = 1.0 - clipped_target_max
                q_values[ x, y] = target_max

                choosen_act[ x, y] = target_argmax

        fig, ax = plt.subplots()
        im = ax.imshow(q_values.T[::-1])

        fig.colorbar(im, ax=ax, label='Interactive colorbar')


        #self.writer.add_image(str(self.agent_id)+"/q_values_imgs", q_values, completed_episodes)


        #fig, ax = plt.subplots(figsize=(6, 6))
        for x in range(env.X_MAX+1):
            for y in range(env.Y_MAX+1):
                if choosen_act[x,y] != 4:
                    plt.arrow(x, env.Y_MAX-y, scale*arrows[choosen_act[x,y]][0], scale*arrows[choosen_act[x,y]][1], head_width=0.1)

        self.writer.add_figure(str(self.agent_id)+"/q*_values_imgs", fig, completed_episodes)

        if self.dueling:
            fig, ax = plt.subplots()
            im = ax.imshow(v_values.T[::-1])

            fig.colorbar(im, ax=ax, label='Interactive colorbar')

            self.writer.add_figure(str(self.agent_id)+"/v_values_imgs", fig, completed_episodes)
 
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
            #target = torch.argmax(considered_q_values).numpy()

            #assert np.all(target >= 0)
            target_max = target.max().float()

            target_argmax = target.argmax()


            #clipped_target_max = (np.clip(target_max, -10, 10) + 10)/ 20
            #q_values[0, x, y] = clipped_target_max 
            #q_values[1, x, y] = 1.0 - clipped_target_max
            q_values[ x, y] = target_max

            choosen_act[ x, y] = target_argmax
            actions.append(target_argmax)

        n_previous_action_mask = n_action_mask
        n_next_obs, n_reward, n_terminated, info = env.step(actions)
        
        n_action_mask = info['avail_actions']

        n_obs = n_next_obs
        #n_previous_action_mask = [env.get_avail_agent_actions(agent_id) for agent_id in range(env.n_agents)]
        state = env.get_state()[:-1]
        terminated = n_terminated[0] or (state in states)
        states.append(state)


    fig, ax = plt.subplots()
    im = ax.imshow(q_values.T[::-1])

    fig.colorbar(im, ax=ax, label='Interactive colorbar')




    #fig, ax = plt.subplots(figsize=(6, 6))
    for x in range(env.X_MAX+1):
        for y in range(env.Y_MAX+1):
            if choosen_act[x,y] != 4:
                plt.arrow(x, env.Y_MAX-y, scale*arrows[choosen_act[x,y]][0], scale*arrows[choosen_act[x,y]][1], head_width=0.1) #, color=

    agents[0].writer.add_figure("q_values_imgs", fig, completed_episodes)


def run_episode(env, q_agents, completed_episodes, params, replay_buffer=None, smaller_buffer=None, training=False, visualisation=False, verbose=False):
    if training:
        assert replay_buffer is not None
        if params['rb'] == 'laber':
            assert smaller_buffer is not None

    
    if visualisation:
        visualize_trajectory(env, q_agents, completed_episodes)
        #for agent in q_agents:
        #    agent.visualize_q_values(env, completed_episodes)

    n_obs, info = env.reset(return_info=True)
    n_action_mask = info['avail_actions']
    n_agents = len(q_agents)
    #n_previous_action_mask = [env.get_avail_agent_actions(agent_id) for agent_id in range(env.n_agents)]
    terminated = False
    episode_reward = 0
    nb_steps = 0
    n_next_obs = None
    #n_action_mask = n_previous_action_mask
    n_previous_action_mask = None
    epsilon = linear_schedule(params['start_e'], params['end_e'], params['exploration_fraction'] * params['total_timesteps'], completed_episodes)


    n_obs = torch.tensor(n_obs)
    #n_next_obs = torch.tensor(n_next_obs)

    n_act_randomly = [params['random_policy'] or (random.random() < epsilon and training) for _ in range(env.n_agents)]
    n_other_act_randomly = torch.tensor([n_act_randomly[:agent_id]+n_act_randomly[agent_id+1:] for agent_id in range(n_agents)])

    if params['add_epsilon']:
        #for a in obs:
        n_epsilon = torch.full((n_agents,1), epsilon)
        n_obs = torch.cat((n_obs, n_epsilon), dim=1).float()
        #n_next_obs = torch.cat((n_next_obs, n_epsilon), dim=1)
    if params['add_others_explo']:
        n_obs = torch.cat((n_obs, n_other_act_randomly.float()), 1).float()
    

    while not terminated:
        #n_obs = env.get_obs()
        #state = env.get_state()
        # env.render()  # Uncomment for rendering

        n_action = []
        n_probabilities = []
        

        
        
        for agent_id, obs in enumerate(n_obs):
            avail_actions = n_action_mask[agent_id]
            #env.get_avail_agent_actions(agent_id)
            avail_actions_ind = np.nonzero(avail_actions)[0]
            
            
            if n_act_randomly[agent_id]:
                action = np.random.choice(avail_actions_ind).reshape(-1)
                probability = epsilon/sum(avail_actions)
            else:
                action = q_agents[agent_id].act(obs, avail_actions, epsilon, others_explo=n_other_act_randomly[agent_id])
                probability = 1.0-epsilon*(sum(avail_actions)-1)/sum(avail_actions) if training else 1.0

            assert probability > 0.0 , (probability, epsilon)
            
            
            n_action.append(action)
            n_probabilities.append(probability)

        if False:
            print("n_previous_action_mask", n_previous_action_mask)
            print("n_action", n_action)
            #env.render()

        n_previous_action_mask = n_action_mask
        n_action = [int(a) for a in n_action]
        n_next_obs, n_reward, n_terminated, info = env.step(n_action)

        n_act_randomly = [params['random_policy'] or (random.random() < epsilon and training) for _ in range(env.n_agents)]
        n_other_act_randomly = torch.tensor([n_act_randomly[:agent_id]+n_act_randomly[agent_id+1:] for agent_id in range(n_agents)])

        n_next_obs = torch.tensor(n_next_obs)
        #n_next_obs = torch.tensor(n_next_obs)

        if params['add_epsilon']:
            #for a in obs:
            n_epsilon = torch.full((n_agents,1), epsilon)
            n_next_obs = torch.cat((n_next_obs, n_epsilon), dim=1).float()
            #n_next_obs = torch.cat((n_next_obs, n_epsilon), dim=1)
        if params['add_others_explo']:
            n_next_obs = torch.cat((n_next_obs, n_other_act_randomly.float()), 1).float()

        if np.array(n_reward).size ==1:
            n_reward = np.full((n_agents, 1), n_reward)
        if np.array(n_terminated).size ==1:
            n_terminated = np.full((n_agents, 1), n_terminated)

        
        #action = deepcopy(action)
        
        
            #n_next_obs = torch.cat((n_next_obs, torch.tensor(n_other_act_randomly)), 1)

        n_next_obs = torch.tensor(n_next_obs, dtype=torch.float)
        n_action_mask = info['avail_actions']

        if False:
            print("n_next_obs", n_next_obs)
            print("n_reward", n_reward)
            print("n_terminated", n_terminated)
            print("n_action_mask", n_action_mask)
            print()
        #print("n_reward", n_reward)
        episode_reward += np.mean(n_reward)


        if training:
            
            for previous_action_mask, action_mask in zip(n_previous_action_mask, n_action_mask):
                assert sum(previous_action_mask) > 0 
                assert sum(action_mask) > 0 

            transition = {
                'observations':torch.tensor(n_obs, dtype=torch.int64).reshape(n_agents, -1),
                'action_mask':torch.tensor(n_previous_action_mask, dtype=torch.int64).reshape(n_agents, -1),
                'actions':torch.tensor(n_action, dtype=torch.int64).reshape(n_agents, -1),
                'actions_likelihood':torch.tensor(n_probabilities, dtype=torch.float).reshape(n_agents, -1),
                'rewards':torch.tensor(n_reward, dtype=torch.float).reshape(n_agents, -1),
                'next_observations':torch.tensor(n_next_obs, dtype=torch.int64).reshape(n_agents, -1),
                'next_action_mask':torch.tensor(n_action_mask, dtype=torch.int64).reshape(n_agents, -1),
                'dones':torch.tensor(n_terminated, dtype=torch.float).reshape(n_agents, -1),
                #'td': 1.0#{a:1.0 for a in obs}
            }

            transition = TensorDict(transition,batch_size=[n_agents])
            replay_buffer.add(transition)


        if verbose:
            print("actions:", actions)
            print("probabilities:", probabilities)
            print("next_obs:", n_next_obs)
            print("reward:", reward)

        nb_steps += 1
        n_obs = n_next_obs
        terminated = n_terminated[0] or nb_steps > params['t_max']

    if training and completed_episodes > params['learning_starts'] and completed_episodes % params['train_frequency'] == 0:
        
        if params['rb'] =='laber':
            # On met a jour les TD errors 
            big_sample = replay_buffer.sample()
            big_sample = add_ratios(big_sample, q_agents, epsilon, params['single_agent'])
            smaller_buffer.extend(big_sample)
            sample = smaller_buffer.sample()
            if params['prioritize_big_buffer']:
                replay_buffer.update_tensordict_priority(big_sample)

        elif params['rb'] =='likely':
            big_sample = replay_buffer.sample()
            big_sample = add_ratios(big_sample, q_agents, epsilon, params['single_agent'])
            sample = torch.stack([get_n_likeliest(big_sample[:,agent_id], params['filter'], params['batch_size']) for agent_id in range(n_agents)], dim=1)
            if params['prioritize_big_buffer']:
                replay_buffer.update_tensordict_priority(big_sample)

        elif params['rb'] == 'correction':
            #samples = get_correclty_sampled_transitions(q_agents, epsilon, nb_transitions, replay_buffer)
            nb_states, nb_transitions = params['batch_size'] // 10, 10
            samples = collate_n_state_batch(nb_states, nb_transitions, q_agents, epsilon, replay_buffer)
            samples = samples.reshape([params['batch_size']] + list(samples.shape[2:]))

            samples = torch.permute(samples, [1, 0, 2])

            #print("returned sample", samples)
        else:
            sample = replay_buffer.sample()
            sample = add_ratios(sample, q_agents, epsilon, params['single_agent'])
        
        if params['rb'] != 'correction':
            samples = [sample for _ in range(n_agents)]
        td_errors = []
        for agent_id, (agent, sample) in enumerate(zip(q_agents, samples)):
            
            agent_td_error = agent.train(sample[:,agent_id], completed_episodes)
            td_errors.append(agent_td_error)
        #print(np.array(td_errors).shape)

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

    return nb_steps, episode_reward #episodic_returns
    
def run_training(env_id, verbose=True, run_name='', path=None, **args):
    with open(Path('src/config/'+env_id+'/default.yaml')) as f:
        params = yaml.safe_load(f)
    
    if path is None:
        path = Path.cwd() / 'results'
    
    for k, v in args.items():
        if v is not None:
            #print('setting', k, 'to', v)
            params[k] = v
    #params.update(args)
    if __name__ == "__main__":
        pprint(params)
    old_params = copy(params)

    #experiment_hash = str(dict_hash(params))
    if params['device'] == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif params['device'] == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    #env = gym.make(env_id)
    if env_id == 'simultaneous':
        env = SimultaneousEnv(n_agents=params['n_agents'], n_actions=params['n_actions'])
    elif env_id == 'water-bomber':
        env = WaterBomberEnv(x_max=params['x_max'], y_max=params['y_max'], t_max=params['t_max'], n_agents=params['n_agents'], obs_normalization=params['env_normalization'], deterministic=params['deterministic_env'], add_id=params['add_id'])
    elif env_id == 'smac':
        #env = gym.make(f"smaclite/MMM2-v0")
        env = gym.make(f"smaclite/3s_vs_5z")
        
    else:
        raise NameError('Unknown env:'+env_id)

    #if params['run_name'] is None:
    #    params['run_name'] = f"iql_{int(time.time())}"
    

    writer = SummaryWriter(path/"runs"/run_name)
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
    
    #env = SimultaneousEnv(n_agents=params['n_agents'], n_actions=10)
    #WaterBomberEnv(x_max=params['x_max'], y_max=params['y_max'], t_max=params['t_max'], n_agents=params['n_agents'])
    # env = dtype_v0(rps_v2.env(), np.float)
    #api_test(env, num_cycles=1000, verbose_progress=True)

    env.reset()

    try:
        obs_shape = env.observation_space[-1].shape
    except:
        obs_shape = env.observation_space(0).shape
        
    size_obs = int(np.product(obs_shape))
    
    try:
        size_act = int(env.action_space[-1].n)
    except:
        size_act = int(env.action_space(0).n)
    
    if verbose:
        print('-'*20)
        print('num_agents: ',env.n_agents)
        #print('observation_space: ',env.observation_space[0])
        #print('action_space: ',env.action_space[0])
        #print('infos: ',env.infos)    
        print('size_obs: ',size_obs)    
        print('size_act: ',size_act)    
        print('-'*20)

    if params['add_epsilon']:
        size_obs += 1
    if params['add_others_explo']:
        size_obs += env.n_agents - 1
    #print("Size obs:", size_obs)

    replay_buffer, smaller_buffer = create_rb(rb_type=params['rb'], buffer_size=params['buffer_size'], batch_size=params['batch_size'], n_agents=env.n_agents, device=params['device'], prio=params['prio'], prioritize_big_buffer=params['prioritize_big_buffer'])


    ### Creating Agents

    q_agents = [QAgent(env, a, params, size_obs, size_act, writer, run_name)  for a in range(env.n_agents)]
    
    if params['load_agents_from'] is not None:
        for name, agent in enumerate(q_agents):
            model_path = f"runs/{params['load_agents_from']}/saved_models/{name}.cleanrl_model"
            agent.load(model_path)
            
    if params['load_buffer_from'] is not None:
        for name, agent in enumerate(q_agents):
            buffer_path = f"runs/{params['load_buffer_from']}/saved_models/{name}_buffer.pkl"
            agent.load_buffer(buffer_path)

    if params['single_agent']:
        agent_0 = q_agents[0]
        for agent in range(env.n_agents):
            q_agents[agent].q_network = agent_0.q_network
            
            assert q_agents[agent].q_network is agent_0.q_network
            #q_agents[agent].replay_buffer = agent_0.replay_buffer

    #with contextlib.suppress(Exception):

    #env_info = env.get_env_info()

    #n_actions = env_info["n_actions"]
    #n_agents = env_info["n_agents"]
        
    results = []
    pbar=trange(params['total_timesteps'])
    for completed_episodes in pbar:
        if not params['no_training']:
            run_episode(env, q_agents, completed_episodes, params, replay_buffer=replay_buffer, smaller_buffer=smaller_buffer, training=True, visualisation=False, verbose=False)
               

        if completed_episodes % params['evaluation_frequency'] == 0:
            if params['save_imgs']:
                run_episode(env, q_agents, completed_episodes, params, training=False, visualisation=True)
            
            list_total_reward = []
            average_duration = 0.0

            for _ in range(params['evaluation_episodes']):

                nb_steps, total_reward = run_episode(env, q_agents, completed_episodes, params, training=False)
                list_total_reward.append(total_reward)
                average_duration += nb_steps
            
            average_duration /= params['evaluation_episodes']
            average_return = np.mean(list_total_reward)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("Average Return", average_return, completed_episodes)
            writer.add_scalar("Size replay buffer", len(replay_buffer), completed_episodes)
            #writer.add_scalar("Mean TD_error", mean_td_errors, completed_episodes)     
            
            if params['track']:
                wandb.log({
                        "Average Return":average_return, 
                        "Completed Episodes:":completed_episodes
                    })
            #writer.add_scalar("Average duration", average_duration, completed_episodes)
            pbar.set_description(f"Return={average_return:5.1f}") #, Duration={average_duration:5.1f}"
            results.append(average_return)
                
    env.close() 

    if params['save_buffer']:
        for agent in q_agents:
            q_agents[agent].save_rb()


    env.close()
    steps = [i for i in range(0, params['total_timesteps'], params['evaluation_frequency'])]
    
    results_dict = {
            'Average optimality': results,
            'Step': steps,
        }
    result_df = pd.DataFrame(results_dict)

    os.makedirs(path / run_name, exist_ok=True)
    #assert str(dict_hash(params)) == experiment_hash, params
    with open(path/ run_name/'params.yaml', 'w') as f:
        yaml.dump(params, f, default_flow_style=False)

    result_df = result_df.assign(**params)
    #os.makedirs(path / experiment_hash, exist_ok=True)
    result_df.to_csv(path / run_name / 'results.csv', index=False)

    assert params == old_params, (params, old_params)
    assert dict_hash(params) == dict_hash(old_params), (params, old_params)
    return steps, results

def create_rb(rb_type, buffer_size, batch_size, n_agents, device, prio, prioritize_big_buffer=False):
    smaller_buffer = None
    rb_storage = LazyTensorStorage(buffer_size, device='cpu')
    if rb_type == 'uniform' or rb_type == 'correction':
        replay_buffer = TensorDictReplayBuffer(
            #replay_buffer = TensorDictReplayBuffer(
            #storage=ListStorage(buffer_size),
            storage=rb_storage,
            #collate_fn=lambda x: x, 
            #priority_key="td_error",
            batch_size=batch_size,
        )

    elif rb_type == 'prioritized':
        #sampler = PrioritizedSampler(max_capacity=buffer_size, alpha=0.8, beta=1.1)
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha = 1.0,#0.7,
            beta = 1.0,#1.1,
            priority_key="td_error",
            #storage=ListStorage(buffer_size),
            storage=rb_storage,
            #collate_fn=lambda x: x, 
            batch_size=batch_size,
        )
        
    else:
        if prioritize_big_buffer:
            replay_buffer = TensorDictPrioritizedReplayBuffer(
                storage=rb_storage,
                alpha = 1.0,#0.7,
                beta = 1.0,#1.1,
                priority_key="td_error",
                batch_size=4*batch_size,
            )
        else:
            replay_buffer = TensorDictReplayBuffer(
                storage=rb_storage,
                #priority_key="td_error",
                batch_size=4*batch_size,
            )
        

        if rb_type =='laber':
            smaller_buffer_size = 4*batch_size

            smaller_buffer = TensorDictPrioritizedReplayBuffer(
                alpha = 1.0, #0.7,
                beta = 1.0, #1.1,
                priority_key=prio,#"td_error",
                #storage=ListStorage(buffer_size),
                storage=LazyTensorStorage(smaller_buffer_size),
                #collate_fn=lambda x: x, 
                batch_size=batch_size,
            )

    return replay_buffer, smaller_buffer


def current_and_past_others_actions_likelyhood(sample, agents, epsilon, single_agent):
        #current_likelyhood, past_likelyhood = torch.ones(.batch_size), torch.ones(.batch_size)
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

            with torch.no_grad():
                    
                q_values = agent.q_network(obs)

                considered_q_values = q_values + (action_mask-1.0)*9999.0
                best_actions = torch.argmax(considered_q_values, dim=1)#.reshape(1)

                #probability = torch.zeros_like(current_likelyhood)
                actions = actions.squeeze()

                assert torch.all(torch.sum(action_mask, 1) >0)
                mask = (actions == best_actions).float()
                #print('mask:', mask)
                #print(mask*(1.0-epsilon))
                #print((1.0-mask)*epsilon/torch.sum(action_mask, 1))
                #print('action_mask:', action_mask)
                #print(torch.sum(action_mask, 1))
                proba_rd = epsilon/torch.sum(action_mask, 1)
                probability = mask*(1.0-epsilon+proba_rd) + (1.0-mask)*proba_rd
                
                current_likelyhood.append(probability.cpu().numpy())
                past_likelyhood.append(old_probability.squeeze().cpu().numpy()) 

        others_current_likelyhood = torch.tensor(np.prod([current_likelyhood[:n]+current_likelyhood[n+1:] for n in range(len(agents))], axis=1)).cpu()
        others_past_likelyhood = torch.tensor(np.prod([past_likelyhood[:n]+past_likelyhood[n+1:] for n in range(len(agents))], axis=1)).cpu()
        #torch.stack([torch.tensor(np.prod(current_likelyhood[:n]+current_likelyhood[n+1:], axis=1)) for n in range(len(agents))], dim=1)
        #others_past_likelyhood = torch.stack([torch.tensor(np.prod(past_likelyhood[:n]+past_likelyhood[n+1:])) for n in range(len(agents))], dim=1)
        return others_current_likelyhood.T, others_past_likelyhood.T


def add_ratios(sample, agents, epsilon, single_agent, completed_episodes=None, writer=None):
    td_error = torch.stack([agent.get_td_error(sample[:,id_agent]).to('cpu') for id_agent, agent in enumerate(agents)], dim=1)
    current_likelyhood, past_likelyhood = current_and_past_others_actions_likelyhood(sample, agents, epsilon, single_agent)
    #current_likelyhood, past_likelyhood = current_likelyhood.to(device) , past_likelyhood.to(device) 

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



def main(**params):

    #for n_agents in range(1,10):
    #params["n_agents"] = n_agents
    #params["run_name"] = str(n_agents)
    params["run_name"] = "test"
    steps, results = run_training(**params)

    wandb.finish()
        #print("results:", results)
    #print("Average total reward", total_reward / args.total_timesteps)


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
    #params = {
    #    'total_timesteps': 1010
    #}
    #main(**params)
    #main()
