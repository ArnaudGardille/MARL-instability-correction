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
from torchrl.data import LazyTensorStorage, LazyMemmapStorage, ListStorage

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
    parser.add_argument("--save-imgs", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save images of the V or Q* functions")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="smac-v1",
        help="the id of the environment")
    parser.add_argument("--load-agents-from", type=str, default=None,
        help="the experiment from which to load agents.")
    parser.add_argument("--load-buffer-from", type=str, default=None,
        help="the experiment from which to load agents.")
    parser.add_argument("--no-training", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to show the video")
    parser.add_argument("--total-timesteps", type=int, default=100000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=1000000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.9,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--evaluation-frequency", type=int, default=1000)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default= 100, #2**18, #256, #
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=1000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    parser.add_argument("--single-agent", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use a single network for all agents. Identity is the added to observation")
    parser.add_argument("--add-id", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to add agents identity to observation")
    parser.add_argument("--dueling", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use a dueling network architecture.")
    parser.add_argument("--prioritized-rb", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use a prioritized replay buffer.")
    args = parser.parse_args()
    # fmt: on
    #assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    return args


args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
print('Device: ', device)

def load_experiment(run_name, load_buffer=False):
    q_agents = None
    return q_agents

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
writer.add_hparams(vars(args), {})
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
        
        if args.prioritized_rb:
            sampler = PrioritizedSampler(max_capacity=self.buffer_size, alpha=0.8, beta=1.1)
        else:
            sampler = RandomSampler()

        self.rb_storage = LazyMemmapStorage(self.buffer_size)

        self.replay_buffer = TensorDictReplayBuffer(
            #storage=ListStorage(self.buffer_size),
            storage=self.rb_storage,
            sampler=sampler,
            #collate_fn=lambda x: x, 
            priority_key="td_error",
            batch_size=self.batch_size,
            )

    def act(self, dict_obs, completed_episodes, training=True):
        normalized_obs = self.env.normalize_obs(dict_obs)
        #dict_obs = TensorDict(dict_obs,batch_size=[])

        obs, avail_actions = normalized_obs['observation'], normalized_obs['action_mask']

        #assert obs[-2] == self.agent_id
        avail_actions_ind = np.nonzero(avail_actions)[0]
        
        epsilon = linear_schedule(self.start_e, self.end_e, self.exploration_fraction * self.total_timesteps, completed_episodes)

        if training and (random.random() < epsilon):
            actions = int(np.random.choice(avail_actions_ind))
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.Tensor(obs).to(device)).cpu()
                #print(q_values, avail_actions, q_values*avail_actions)
                #considered_q_values = q_values*avail_actions
                considered_q_values = q_values + (avail_actions-1.0)*9999.0

                actions = int(torch.argmax(considered_q_values).numpy())

        avail_actions_ind = np.nonzero(avail_actions)
        assert actions in avail_actions_ind

        if completed_episodes % self.train_frequency == 0:
            writer.add_scalar(self.name+"/epsilon", epsilon, completed_episodes)
            writer.add_scalar(self.name+"/action", actions, completed_episodes)


        return actions

    def train(self, completed_episodes):
        # ALGO LOGIC: training.
        if completed_episodes > self.learning_starts:
            #print("mod: ", (completed_episodes + 100*self.agent_id ) % self.train_frequency)
            if completed_episodes % self.train_frequency == 0:
                sample = self.replay_buffer.sample()
                print('index', sample["index"])
                #print('sample:', sample)
                sample = sample.to(device)
                #action_mask = data.next_observations['action_mask']
                normalized_obs = sample['observations']['observation']
                #normalized_obs = self.env.normalize_obs(observations).to(device)
                action_mask = sample['observations']['action_mask']
                
                normalized_next_obs = sample['next_observations']['observation']
                #normalized_next_obs = self.env.normalize_obs(next_observations).to(device)
                next_action_mask = sample['next_observations']['action_mask']
                #assert next_observations[0][-2] == self.agent_id
                
                with torch.no_grad():
                    target_max, _ = (self.target_network(normalized_next_obs)*next_action_mask).max(dim=1)
                    td_target = sample['rewards'].flatten() + self.gamma * target_max * (1 - sample['dones'].flatten())
                old_val = (self.q_network(normalized_obs)*action_mask).gather(1, sample['actions']).squeeze()

                loss = F.mse_loss(td_target, old_val)
                
                if args.prioritized_rb:
                    sample["td_error"] = torch.abs(td_target-old_val)
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

    def add_to_rb(self, obs, action, reward, next_obs, terminated, truncated=False, infos=None):
        
        #normalized_obs = copy(obs)

        normalized_obs = self.env.normalize_obs(obs)

        #normalized_obs = copy(obs)
        normalized_next_obs = self.env.normalize_obs(next_obs)
        
        transition = {
            'observations':normalized_obs,
            'actions':[action],
            'rewards':reward,
            'next_observations':normalized_next_obs,
            'dones':torch.tensor(terminated or truncated, dtype=torch.float),
            #'infos':infos
        }
        transition = TensorDict(transition,batch_size=[])
        #print('transition:', transition)
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
    
    def save_buffer(self):
        buffer_path = f"runs/{run_name}/saved_models/{self.name}_buffer.pkl"
        save_to_pkl(buffer_path, self.replay_buffer)

    def load_buffer(self, buffer_path):
        self.replay_buffer = load_from_pkl(buffer_path)
        assert isinstance(self.replay_buffer, ReplayBuffer), "The replay buffer must inherit from ReplayBuffer class"

    def visualize_q_values(self, env, completed_episodes):
        arrows = {1:(1,0), 3:(-1,0), 2:(0,1), 0:(0,-1)}

        observation, _ = env.reset()
        observation = observation[self.name]['observation']
        observation[-1] = 5
        #assert observation[-2] == self.agent_id

        q_values = np.zeros((env.X_MAX+1, env.Y_MAX+1))
        if self.dueling:
            v_values = np.zeros((env.X_MAX+1, env.Y_MAX+1))
        choosen_act = np.zeros((env.X_MAX+1, env.Y_MAX+1), dtype=int)
        #q_values = np.zeros((3,env.X_MAX, env.Y_MAX))
        for x in range(env.X_MAX+1):
            for y in range(env.Y_MAX+1):
                observation[4+2*self.agent_id ] = x
                observation[4+2*self.agent_id  + 1] = y
                #print(observation)

                action_mask= env.get_action_mask(x,y)
                normalized_obs = self.env.normalize_obs(observation)
                normalized_obs = normalized_obs.to(device)
                #print(self.q_network(observation).detach().cpu()*action_mask)
                pred = self.q_network(normalized_obs).detach().cpu()
                target = pred + (action_mask-1)*9999.0
                #target = torch.argmax(considered_q_values).numpy()

                #assert np.all(target >= 0)
                target_max = target.max().float()

                target_argmax = target.argmax()

                if self.dueling:
                    v_values[ x, y] = self.q_network(normalized_obs, value_only=True).detach().cpu()

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
        



        

 


def run_episode(env, q_agents, completed_episodes, training=False, visualisation=False, verbose=False):
    if visualisation and args.save_imgs:
        obs, _ = env.reset()

        for agent in env.agents:
            q_agents[agent].visualize_q_values(env, completed_episodes)

    obs, _ = env.reset()
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

        
        actions = {agent: q_agents[agent].act(obs[agent], completed_episodes, training) for agent in env.agents}  

        #actions = {agent: np.random.choice(np.nonzero(obs[agent]['action_mask'])[0]) for agent in env.agents}  
        if verbose:
            print("actions:", actions)
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        if verbose:
            print("next_obs:", next_obs)
            print("rewards:", rewards)

        #if training:
        # On entraine pas, mais on complete quand meme le replay buffer
        for agent in obs:
            if False and rewards[agent]>0:
                print("obs:",obs[agent])
                print("actions:",actions[agent])
                print("rewards:",rewards[agent])
                print("next_obs:",next_obs[agent])
                print("terminations:",terminations[agent])

            q_agents[agent].add_to_rb(obs[agent], actions[agent], rewards[agent], next_obs[agent], terminations[agent], truncations[agent], infos[agent])

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
    env.reset()

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
    env = WaterBomberEnv(x_max=4, y_max=4, t_max=20, n_agents=2, deterministic=False)
    # env = dtype_v0(rps_v2.env(), np.float32)
    #api_test(env, num_cycles=1000, verbose_progress=True)

    env.reset()

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
            nb_steps, total_reward = run_episode(env, q_agents, completed_episodes, training=False, visualisation=True)
            
            list_total_reward = []
            average_duration = 0.0

            for _ in range(args.evaluation_episodes):

                nb_steps, total_reward = run_episode(env, q_agents, completed_episodes, training=False)
                list_total_reward.append(total_reward)
                average_duration += nb_steps
            
            average_duration /= args.evaluation_episodes
            average_return = np.mean(list_total_reward)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("Average return", average_return, completed_episodes)
            writer.add_scalar("Average duration", average_duration, completed_episodes)
            
            pbar.set_description(f"Return={average_return:5.1f}, Duration={average_duration:5.1f}")
            

    for agent in q_agents:
        q_agents[agent].save_buffer()

    env.close()

    #print("Average total reward", total_reward / args.total_timesteps)



if __name__ == "__main__":
    main()
    #test()
