from water_bomber_env import WaterBomberEnv

import random
import numpy as np
from time import sleep
from copy import deepcopy
import pickle
scale = 0.25   

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import os
from pathlib import Path
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

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--display-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to show the video")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")
    parser.add_argument("--use-state", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether we give the global state to agents instead of their respective observation")
    parser.add_argument("--save-buffer", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--load-buffer", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--save-imgs", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save images of the V or Q* functions")
    parser.add_argument("--run-name", type=str, default=None)
    

    # Environment specific arguments
    parser.add_argument("--x-max", type=int, default=4)
    parser.add_argument("--y-max", type=int, default=4)
    parser.add_argument("--t-max", type=int, default=10)
    parser.add_argument("--n-agents", type=int, default=2)
    parser.add_argument("--env-normalization", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="water-bomber-v0",
        help="the id of the environment")
    parser.add_argument("--load-agents-from", type=str, default=None,
        help="the experiment from which to load agents.")
    parser.add_argument("--load-buffer-from", type=str, default=None,
        help="the experiment from which to load agents.")
    parser.add_argument("--random-policy", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--no-training", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to show the video")
    parser.add_argument("--total-timesteps", type=int, default=100000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=1000000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--evaluation-frequency", type=int, default=1000)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default= 1000, #2**18, #256, #
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=1000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=100,
        help="the frequency of training")
    parser.add_argument("--single-agent", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use a single network for all agents. Identity is the added to observation")
    parser.add_argument("--add-id", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to add agents identity to observation")
    parser.add_argument("--add-epsilon", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to add epsilon to observation")
    parser.add_argument("--dueling", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use a dueling network architecture.")
    parser.add_argument("--deterministic-env", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--boltzmann-policy", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--loss-corrected-for-others", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--loss-not-corrected-for-prioritized", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--prio", choices=['td', 'td/past', 'td*cur/past', 'td*cur', 'cur/past', 'cur'], default=None)
    parser.add_argument("--rb", choices=['uniform', 'prioritized', 'laber'], default='uniform',
        help="whether to use a prioritized replay buffer.")
    args = parser.parse_args()
    # fmt: on
    #assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    return args


args = parse_args()

def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).mean()

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
print('Device: ', device)

def load_experiment(run_name, load_buffer=False):
    q_agents = None
    return q_agents

if args.run_name is not None:
    run_name = args.run_name
else:
    run_name = f"iql_{int(time.time())}"

if args.save_model:
    os.makedirs(f"runs/{run_name}/saved_models", exist_ok=True)

if args.track:
    import wandb

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )
writer = SummaryWriter(f"runs/{run_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)
#writer.add_hparams(vars(args), {})
writer.flush()

# TRY NOT TO MODIFY: seeding
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

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

    

class QAgent():
    def __init__(self, env, name, args, obs_shape, act_shape):
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.env = env

        self.agent_id  = int(name[-1])
        self.name = str(name)
        self.action_space = env.action_space(self.name)

        if self.dueling:
            network_class=DuelingQNetwork
        else:
            network_class=QNetwork


        self.q_network = network_class(obs_shape, act_shape).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.target_network = network_class(obs_shape, act_shape).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        #observation_space = env.observation_space(agent_id)
        #print(self.buffer_size,env.observation_space(self.name),env.action_space(self.name),device)
        """observation_space = Dict({
            'observation': env.observation_space(self.name),
            'action_mask': env.observation_space(self.name)['action_mask']
        })"""

        """self.replay_buffer = DictReplayBuffer(
            self.buffer_size,
            env.observation_space(self.name), #env.observation_space(self.name)
            env.action_space(self.name),
            device,handle_timeout_termination=False,
            )"""
        self.rb_storage = LazyTensorStorage(self.buffer_size)
        
        if args.rb == 'prioritized':
            #sampler = PrioritizedSampler(max_capacity=self.buffer_size, alpha=0.8, beta=1.1)
            self.replay_buffer = TensorDictPrioritizedReplayBuffer(
                alpha = 1.0,#0.7,
                beta = 1.0,#1.1,
                priority_key="td_error",
                #storage=ListStorage(self.buffer_size),
                storage=self.rb_storage,
                #collate_fn=lambda x: x, 
                batch_size=self.batch_size,
            )
            
        else:
            self.replay_buffer = TensorDictReplayBuffer(
                #self.replay_buffer = TensorDictReplayBuffer(
                #storage=ListStorage(self.buffer_size),
                storage=self.rb_storage,
                #collate_fn=lambda x: x, 
                #priority_key="td_error",
                batch_size=self.batch_size,
            )

            if args.rb =='laber':
                self.smaller_buffer_size = self.batch_size*4

                self.smaller_buffer = TensorDictPrioritizedReplayBuffer(
                    alpha = 1.0, #0.7,
                    beta = 1.0, #1.1,
                    priority_key=self.prio,#"td_error",
                    #storage=ListStorage(self.buffer_size),
                    storage=LazyTensorStorage(self.smaller_buffer_size),
                    #collate_fn=lambda x: x, 
                    batch_size=self.batch_size,
            )
        
        if args.load_buffer:
            self.load_rb()



    def act(self, dict_obs, completed_episodes, training=True):
        #print(dict_obs)
        normalized_obs = self.env.normalize_obs(dict_obs)
        #dict_obs = TensorDict(dict_obs,batch_size=[])

        obs, avail_actions = normalized_obs['observation'], normalized_obs['action_mask']

        #assert obs[-2] == self.agent_id
        assert sum(avail_actions)>0, avail_actions
        avail_actions_ind = np.nonzero(avail_actions).reshape(-1)
        #print('avail_actions_ind', avail_actions_ind)
        
        epsilon = linear_schedule(self.start_e, self.end_e, self.exploration_fraction * self.total_timesteps, completed_episodes)

        

        if args.random_policy or (training and not self.boltzmann_policy and (random.random() < epsilon)):
        #if True:
            action = torch.tensor([np.random.choice(avail_actions_ind)])
            #print(avail_actions_ind, action)
            probability = epsilon/sum(avail_actions)
        else:
            with torch.no_grad():
                obs = torch.Tensor(obs)
                if args.add_epsilon:
                    obs = torch.cat((obs, torch.tensor([epsilon])), 0)
                q_values = self.q_network(obs.to(device)).cpu()

                if self.boltzmann_policy:
                    tres = torch.nn.Threshold(0.001, 0.001)
                    probabilities = tres(q_values)*avail_actions
                    min_q_value = torch.min(q_values + (1.0-avail_actions)*9999.0)
                    probabilities -= probabilities.min()
                    probabilities /= probabilities.sum()

                    action = torch.multinomial(probabilities, 1)
                    probability = probabilities[action]
                else:
                    assert sum(avail_actions)>0, avail_actions
                    considered_q_values = q_values + (avail_actions-1.0)*9999.0
                    #print(considered_q_values)
                    action = torch.argmax(considered_q_values).reshape(1)
                    #print(action)
                    probability = 1.0-epsilon if training else 1.0

        assert probability > 0.0 , (probability, epsilon)
        #assert action in avail_actions_ind
        if action not in avail_actions_ind:
            #print(avail_actions_ind)
            action = torch.tensor([np.random.choice(avail_actions_ind)])
            probability = epsilon/sum(avail_actions)

        if completed_episodes % 1000 == 0:
            writer.add_scalar(self.name+"/epsilon", epsilon, completed_episodes)
            writer.add_scalar(self.name+"/action", action, completed_episodes)

        return action, probability

    def train(self, completed_episodes):
        # ALGO LOGIC: training.
        if completed_episodes > self.learning_starts:
            #print("mod: ", (completed_episodes + 100*self.agent_id ) % self.train_frequency)
            if completed_episodes % self.train_frequency == 0:
                if args.rb =='laber':
                    # On met a jour les TD errors 
                    for _ in range(4): #(self.smaller_buffer_size // self.buffer_size)+1):
                        sample = self.replay_buffer.sample().to(device)
                        
                        td_error = self.get_td_error(sample)
                        current_likelyhood, past_likelyhood = self.current_and_past_others_actions_likelyhood(sample, completed_episodes)
                        current_likelyhood, past_likelyhood = current_likelyhood.to(device) , past_likelyhood.to(device) 

                        sample.set("td_error",td_error)
                        sample.set("td/past",td_error/past_likelyhood)
                        sample.set("td*cur/past",td_error*current_likelyhood/past_likelyhood)
                        sample.set("td*cur",td_error*current_likelyhood)
                        sample.set("cur/past",current_likelyhood/past_likelyhood)
                        sample.set("cur",current_likelyhood)

                        self.smaller_buffer.extend(sample)

                    sample = self.smaller_buffer.sample()
                else:
                    sample = self.replay_buffer.sample()
                
                #if args.rb == 'prioritized':
                #    print('index', sample["index"])
                #print('sample:', sample)

                sample = sample.to(device)
                #action_mask = data.next_observations['action_mask']
                normalized_obs = sample['observations'][self.name]['observation']
                #normalized_obs = self.env.normalize_obs(observations).to(device)
                action_mask = sample['observations'][self.name]['action_mask']
                
                normalized_next_obs = sample['next_observations'][self.name]['observation']
                #normalized_next_obs = self.env.normalize_obs(next_observations).to(device)
                next_action_mask = sample['next_observations'][self.name]['action_mask']
                #assert next_observations[0][-2] == self.agent_id
                assert torch.all(torch.sum(action_mask, 1) >0), (normalized_obs,action_mask)
                assert torch.all(torch.sum(next_action_mask, 1) >0), action_mask
                
                with torch.no_grad():
                    target_max, _ = (self.target_network(normalized_next_obs)*next_action_mask).max(dim=1)
                    #print(sample['rewards'][self.name].shape, target_max.shape,  sample['dones'].shape)
                    td_target = sample['rewards'][self.name].flatten() + self.gamma * target_max * (1 - sample['dones'][self.name].flatten())
                #print(self.q_network(normalized_obs).shape, action_mask.shape, sample['actions'][self.name].shape)
                #old_val = (self.q_network(normalized_obs)*action_mask).gather(1, sample['actions'][self.name].unsqueeze(0)).squeeze()
                old_val = (self.q_network(normalized_obs)*action_mask).gather(1, sample['actions'][self.name]).squeeze()

                weights = torch.ones(self.batch_size)
                
                if args.rb == 'prioritized':
                    with torch.no_grad():
                        td_error = torch.abs(td_target-old_val)
                elif args.rb == 'laber':
                    td_error = sample['args.prio']


                if (args.rb != 'uniform') and (not self.loss_not_corrected_for_prioritized):
                    priorities = self.compute_priorities(td_error)
                    weights *= self.compute_prioritized_correction(priorities)
                    #sample['_weight']

                if self.loss_corrected_for_others:
                    weights *= self.importance_weight(sample, completed_episodes)
                    #print('Shapes:',td_target.shape, old_val.shape, weight.shape)
                loss = weighted_mse_loss(td_target, old_val, weights)
                #else:
                #    loss = F.mse_loss(td_target, old_val)

                if args.rb == 'prioritized':
                    sample.set("td_error",td_error)
                    self.replay_buffer.update_tensordict_priority(sample)

                writer.add_scalar(self.name+"/td_loss", loss, completed_episodes)
                writer.add_scalar(self.name+"/q_values", old_val.mean().item(), completed_episodes)
                writer.add_scalar(self.name+"/size replay buffer", len(self.replay_buffer), completed_episodes)

                # optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.save_model:
                    self.save()

            # update target network
            if completed_episodes % self.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                    target_network_param.data.copy_(
                        self.tau * q_network_param.data + (1.0 - self.tau) * target_network_param.data
                    )

        

        if self.upload_model:
            self.upload_model()

    def compute_priorities(self, td_errors, eps=1e-8):
        return torch.maximum(td_errors/td_errors.sum(), torch.ones(self.batch_size)*eps)
    
    def compute_prioritized_correction(self, priorities):
        return 1.0/priorities
        

    def add_to_rb(self, obs, action, probabilities, reward, next_obs, terminated, truncated=False, infos=None, completed_episodes=0):
        
        obs = deepcopy(obs)
        next_obs = deepcopy(next_obs)
        normalized_obs, normalized_next_obs, dones = {}, {}, {}
        for a in obs:
            if args.env_normalization:
                normalized_obs[a] = self.env.normalize_obs(obs[a])
                normalized_next_obs[a] = self.env.normalize_obs(next_obs[a])
            else:
                normalized_obs[a] = obs[a]
                normalized_next_obs[a] = next_obs[a]
            dones[a] = torch.tensor(terminated[a] or truncated[a], dtype=torch.float)

        if args.add_epsilon:
            for a in obs:
                epsilon = linear_schedule(self.start_e, self.end_e, self.exploration_fraction * self.total_timesteps, completed_episodes)
                normalized_obs[a]['observation'] = torch.cat((normalized_obs[a]['observation'], torch.tensor([epsilon])), 0)
                normalized_next_obs[a]['observation'] = torch.cat((normalized_next_obs[a]['observation'], torch.tensor([epsilon])), 0)

        #normalized_obs = self.env.normalize_obs(obs)
        for a in obs:
            assert torch.sum(obs[a]['action_mask']) > 0 
            assert torch.sum(next_obs[a]['action_mask']) > 0 
        #normalized_next_obs = self.env.normalize_obs(next_obs)

        transition = {
            'observations':normalized_obs,
            'actions':action,
            'actions_likelihood':probabilities,
            'rewards':reward,
            'next_observations':normalized_next_obs,
            'dones':dones,
            #'infos':infos
        }
        
        #print('transition:', transition)
        transition = TensorDict(transition,batch_size=[])
        for a in obs:
            #print('-'*20)
            #print(a, transition['observations'])
            assert torch.sum(transition['observations'][a]['action_mask']) > 0, transition['observations'][a]['action_mask']
            assert torch.sum(transition['next_observations'][a]['action_mask']) > 0, transition['next_observations'][a]['action_mask']
        self.replay_buffer.add(transition)


    def save(self):

        model_path = f"runs/{run_name}/saved_models/{self.name}.cleanrl_model"
        torch.save(self.q_network.state_dict(), model_path)
        #print(f"model saved to {model_path}")

    def load(self, model_path):
        self.q_network.load_state_dict(torch.load(model_path))
        print(f"model sucessfullt loaded from to {model_path}")
        
    def __str__(self):
        pprint(self.__dict__)
        return ""
    
    def save_rb(self):
        #buffer_path = f"runs/{run_name}/saved_models/{self.name}_buffer.pkl"
        #save_to_pkl(buffer_path, self.replay_buffer)
        env_type = "_det" if self.deterministic_env else "_rd"
        add_eps = "_eps" if self.add_epsilon else ""
        path = 'rbs' / Path(self.name+env_type+add_eps+'_replay_buffer.pickle')
        with open(path, 'wb') as handle:
            pickle.dump(self.replay_buffer[:], handle)
        
        #self.rb_storage[:]

    def load_rb(self):
        #self.replay_buffer = load_from_pkl(buffer_path)
        #assert isinstance(self.replay_buffer, ReplayBuffer), "The replay buffer must inherit from ReplayBuffer class"
        env_type = "_det" if self.deterministic_env else "_rd"
        add_eps = "_eps" if self.add_epsilon else ""

        path = 'rbs' / Path(self.name+env_type+add_eps+'_replay_buffer.pickle')
        with open(path, 'rb') as handle:
            data = pickle.load(handle)

        self.replay_buffer.extend(data)

    def get_td_error(self, sample):
        sample = sample.to(device)
        normalized_obs = sample['observations'][self.name]['observation']
        action_mask = sample['observations'][self.name]['action_mask']
        normalized_next_obs = sample['next_observations'][self.name]['observation']
        next_action_mask = sample['next_observations'][self.name]['action_mask']
        
        with torch.no_grad():
            target_max, _ = (self.target_network(normalized_next_obs)*next_action_mask).max(dim=1)
            #print(sample['rewards'][self.name].shape, target_max.shape,  sample['dones'][self.name].shape)
            td_target = sample['rewards'][self.name].flatten() + self.gamma * target_max * (1 - sample['dones'][self.name].flatten())
            old_val = (self.q_network(normalized_obs)*action_mask).gather(1, sample['actions'][self.name]).squeeze()

            td_error = torch.abs(td_target-old_val)
        return td_error

    def visualize_q_values(self, env, completed_episodes):
        arrows = {1:(1,0), 3:(-1,0), 2:(0,1), 0:(0,-1)}

        observation, _ = env.reset(deterministic=True)
        observation = observation[self.name] #['observation']
        #observation['observation'][-1] = 5
        #assert observation[-2] == self.agent_id

        q_values = np.zeros((env.X_MAX+1, env.Y_MAX+1))
        if self.dueling:
            v_values = np.zeros((env.X_MAX+1, env.Y_MAX+1))
        choosen_act = np.zeros((env.X_MAX+1, env.Y_MAX+1), dtype=int)
        #q_values = np.zeros((3,env.X_MAX, env.Y_MAX))
        for x in range(env.X_MAX+1):
            for y in range(env.Y_MAX+1):
                #observation['observation'][4+2*self.agent_id ] = x
                #observation['observation'][4+2*self.agent_id  + 1] = y
                #print(observation)

                action_mask= env.get_action_mask(x,y)
                normalized_obs = self.env.normalize_obs(observation)
                #print('normalized_obs:', normalized_obs)
                normalized_obs = TensorDict(normalized_obs, batch_size=[]).to(device)
                #print(self.q_network(observation).detach().cpu()*action_mask)
                pred = self.q_network(normalized_obs['observation']).detach().cpu()
                target = pred + (action_mask-1)*9999.0
                #target = torch.argmax(considered_q_values).numpy()

                #assert np.all(target >= 0)
                target_max = target.max().float()

                target_argmax = target.argmax()

                if self.dueling:
                    v_values[ x, y] = self.q_network(normalized_obs['observation'], value_only=True).detach().cpu()

                #clipped_target_max = (np.clip(target_max, -10, 10) + 10)/ 20
                #q_values[0, x, y] = clipped_target_max 
                #q_values[1, x, y] = 1.0 - clipped_target_max
                q_values[ x, y] = target_max

                choosen_act[ x, y] = target_argmax

        fig, ax = plt.subplots()
        im = ax.imshow(q_values.T[::-1])

        fig.colorbar(im, ax=ax, label='Interactive colorbar')


        #writer.add_image(self.name+"/q_values_imgs", q_values, completed_episodes)


        #fig, ax = plt.subplots(figsize=(6, 6))
        for x in range(env.X_MAX+1):
            for y in range(env.Y_MAX+1):
                if choosen_act[x,y] != 4:
                    plt.arrow(x, 4-y, scale*arrows[choosen_act[x,y]][0], scale*arrows[choosen_act[x,y]][1], head_width=0.1)

        writer.add_figure(self.name+"/q*_values_imgs", fig, completed_episodes)

        if self.dueling:
            fig, ax = plt.subplots()
            im = ax.imshow(v_values.T[::-1])

            fig.colorbar(im, ax=ax, label='Interactive colorbar')

            writer.add_figure(self.name+"/v_values_imgs", fig, completed_episodes)
        
    def importance_weight(self, sample, completed_episodes):
        num, denom = self.current_and_past_others_actions_likelyhood(sample, completed_episodes)
        return (num/denom).to(device)
    
    def current_and_past_others_actions_likelyhood(self, sample, completed_episodes):
        current_likelyhood, past_likelyhood = torch.ones(self.batch_size), torch.ones(self.batch_size)

        for agent in self.env.possible_agents:
            if agent != self.name:
                sample = sample.cpu() #.to(device)
                normalized_obs = sample['observations'][agent]['observation']
                action_mask = sample['observations'][agent]['action_mask']
                actions = sample['actions'][agent]

                with torch.no_grad():
                        
                    q_values = self.q_network(torch.Tensor(normalized_obs).to(device)).cpu()

                    if self.boltzmann_policy:
                        
                        tres = torch.nn.Threshold(0.001, 0.001)
                        probabilities = tres(q_values)*action_mask
                        min_q_value = torch.min(q_values + (1.0-action_mask)*9999.0)
                        probabilities -= probabilities.min()
                        probabilities /= probabilities.sum()
                        
                        #print("actions:", actions.shape)
                        #print("probabilities:", probabilities.shape)
                        probability = probabilities.gather(1, sample['actions'][self.name]).squeeze()
                        #print("probability:", probability.shape)
                        current_likelyhood *= probability
                    else:
                        epsilon = linear_schedule(self.start_e, self.end_e, self.exploration_fraction * self.total_timesteps, completed_episodes)
                        considered_q_values = q_values + (action_mask-1.0)*9999.0
                        best_actions = torch.argmax(considered_q_values, dim=1)#.reshape(1)

                        probability = torch.zeros(self.batch_size)
                        actions = actions.squeeze()

                        assert torch.all(torch.sum(action_mask, 1) >0)
                        mask = (actions == best_actions).float()
                        #print('mask:', mask)
                        #print(mask*(1.0-epsilon))
                        #print((1.0-mask)*epsilon/torch.sum(action_mask, 1))
                        #print('action_mask:', action_mask)
                        #print(torch.sum(action_mask, 1))
                        probability = mask*(1.0-epsilon) + (1.0-mask)*epsilon/torch.sum(action_mask, 1)
                        current_likelyhood *= probability
                     
                    past_likelyhood *= sample['actions_likelihood'][agent].squeeze()

        #print("current_likelyhood:", current_likelyhood) #shape
        #print("past_likelyhood:", past_likelyhood)
        #print("ratio:", (current_likelyhood/past_likelyhood).shape)
        return current_likelyhood, past_likelyhood
                
                    
                        


                

 


def run_episode(env, q_agents, completed_episodes, training=False, visualisation=False, verbose=False, deterministic=args.deterministic_env):
    if visualisation and args.save_imgs:
        obs, _ = env.reset(deterministic=deterministic) 

        for agent in env.agents:
            q_agents[agent].visualize_q_values(env, completed_episodes)

    obs, _ = env.reset(deterministic=deterministic)
    optimal_reward = env.compute_optimal_reward()

    if verbose:
        print('initial obs:', obs)

    #episodic_returns = {}
    episodic_return = 0.0
    nb_steps = 0

    while env.agents:
        if visualisation:
            env.render()
            time.sleep(0.1)
        
        #if args.use_state:
        #    for a in env.agents:
        #        obs[a]['observation'] = env.state()

        actions, probabilities = {}, {}
        for agent in env.agents:
            action, prbability = q_agents[agent].act(obs[agent], completed_episodes, training)
            actions[agent] = action
            probabilities[agent] = prbability

        #actions = {agent: np.random.choice(np.nonzero(obs[agent]['action_mask'])[0]) for agent in env.agents}  
        if verbose:
            print("actions:", actions)
            print("probabilities:", probabilities)
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        if verbose:
            print("next_obs:", next_obs)
            print("rewards:", rewards)

        if training:
        # On entraine pas, mais on complete quand meme le replay buffer
            for agent in obs:
                #q_agents[agent].add_to_rb(obs[agent], actions[agent], rewards[agent], next_obs[agent], terminations[agent], truncations[agent], infos[agent])
                q_agents[agent].add_to_rb(obs, actions, probabilities, rewards, next_obs, terminations, truncations, infos, completed_episodes=completed_episodes)

        #episodic_returns = {k: rewards.get(k, 0) + episodic_returns.get(k, 0) for k in set(rewards) | set(episodic_returns)}
        episodic_return += np.mean(list(rewards.values())) 

        if visualisation and False:
            print('Masks:', {a:obs[a]['action_mask'] for a in obs})
            print('Actions:', actions)
            print("Return:",episodic_return)

        #assert episodic_return < 0.0
        
        nb_steps += 1

        obs = next_obs

    if training:
        if args.single_agent:
            agent_0 = list(q_agents.values())[0]
            agent_0.train(completed_episodes)
        else:
            for agent in q_agents.values():
                agent.train(completed_episodes)

    return nb_steps, episodic_return/optimal_reward #episodic_returns
    
def test():
    env = simple_spread_v3.env(N=2)
    #simple_v3.env()
    env.reset(deterministic=args.deterministic_env)

    agent_0 = env.agents[0]

    print(env.observation_space(agent_0))
    obs_shape = env.observation_space(agent_0).shape
    size_obs = np.product(obs_shape)
    size_act = int(env.action_space(agent_0).n)

    print('-'*20)
    print('agents: ',env.agents)
    print('num_agents: ',env.num_agents)
    print('observation_space: ', env.observation_space(agent_0))
    print('action_space: ',env.action_space(agent_0))
    #print('infos: ',env.infos)    
    print('size_obs: ',size_obs)    
    print('size_act: ',size_act)    
    print('-'*20)

    print(env.action_space(agent_0).sample())
    print(agent_0)
    #print(env.step(1)) #{agent_0: 1}))
    q_agents = {a:QAgent(env, a, args, size_obs, size_act)  for a in env.agents}



def main():

    ### Creating Env
    env = WaterBomberEnv(x_max=args.x_max, y_max=args.y_max, t_max=args.t_max, n_agents=args.n_agents)
    # env = dtype_v0(rps_v2.env(), np.float32)
    #api_test(env, num_cycles=1000, verbose_progress=True)

    env.reset(deterministic=args.deterministic_env)

    agent_0 = env.agents[0]
    print(env.observation_space(agent_0))
    obs_shape = env.observation_space(agent_0)['observation'].shape
    size_obs = np.product(obs_shape)
    
    size_act = int(env.action_space(agent_0).n)
    
    
    print('-'*20)
    print('agents: ',env.agents)
    print('num_agents: ',env.num_agents)
    print('observation_space: ',env.observation_space(agent_0))
    print('action_space: ',env.action_space(agent_0))
    #print('infos: ',env.infos)    
    print('size_obs: ',size_obs)    
    print('size_act: ',size_act)    
    print('-'*20)

    if args.add_epsilon:
        size_obs += 1
    
    ### Creating Agents
    
    q_agents = {a:QAgent(env, a, args, size_obs, size_act)  for a in env.agents} 
    
    if args.load_agents_from is not None:
        for name, agent in q_agents.items():
            model_path = f"runs/{args.load_agents_from}/saved_models/{name}.cleanrl_model"
            agent.load(model_path)
            
    if args.load_buffer_from is not None:
        for name, agent in q_agents.items():
            buffer_path = f"runs/{args.load_buffer_from}/saved_models/{name}_buffer.pkl"
            agent.load_buffer(buffer_path)

    if args.single_agent:
        agent_0 = q_agents[env.agents[0]]
        for agent in q_agents:
            q_agents[agent].q_network = agent_0.q_network
            q_agents[agent].replay_buffer = agent_0.replay_buffer


    pbar=trange(args.total_timesteps)
    for completed_episodes in pbar:
        if not args.no_training:
            run_episode(env, q_agents, completed_episodes, training=True)


        if completed_episodes % args.evaluation_frequency == 0:
            if args.display_video:
                    nb_steps, total_reward = run_episode(env, q_agents, completed_episodes, training=False, visualisation=True)
            
            determinims = [False] 
            determinims += [True] if (args.x_max==4 and args.y_max==4 and args.t_max==20 and args.n_agents==2) else []
            for deterministic in determinims:
                list_total_reward = []
                average_duration = 0.0

                for _ in range(args.evaluation_episodes):

                    nb_steps, total_reward = run_episode(env, q_agents, completed_episodes, training=False, deterministic=deterministic)
                    list_total_reward.append(total_reward)
                    average_duration += nb_steps
                
                average_duration /= args.evaluation_episodes
                average_return = np.mean(list_total_reward)

                # TRY NOT TO MODIFY: record rewards for plotting purposes
                decr = "Average return " + ("deterministic" if deterministic else "stochastic")
                writer.add_scalar(decr, average_return, completed_episodes)
                #writer.add_scalar("Average duration", average_duration, completed_episodes)
                if not deterministic:
                    pbar.set_description(f"Return={average_return:5.1f}") #, Duration={average_duration:5.1f}"
                

    if args.save_buffer:
        for agent in q_agents:
            q_agents[agent].save_rb()

    env.close()

    #print("Average total reward", total_reward / args.total_timesteps)



if __name__ == "__main__":
    main()
    #test()
