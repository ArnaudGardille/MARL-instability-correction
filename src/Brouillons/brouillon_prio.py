
#%%
from torchrl.data import TensorDictReplayBuffer
from torchrl.data import LazyMemmapStorage, LazyTensorStorage, ListStorage
from tensordict import tensorclass, TensorDict
from torchrl.data.replay_buffers.samplers import PrioritizedSampler

import torch

#%%
# We define the maximum size of the buffer
size = 10_000

data = TensorDict(
    {
        "a": (torch.arange(12)+1).view(3, 4),
        ("b", "c"): torch.arange(15).view(3, 5),
    },
    batch_size=[3],
)
print(data)
#%%
rb_storage = LazyTensorStorage(size, scratch_dir="~/Documents/Dassault/Water-Bomber-Env/src/memmap/")


buffer_lazymemmap = TensorDictReplayBuffer(
        #self.replay_buffer = TensorDictReplayBuffer(
        #storage=ListStorage(self.buffer_size),
        storage=rb_storage,
        #collate_fn=lambda x: x, 
        sampler=PrioritizedSampler(max_capacity=size, alpha=0.8, beta=1.1),
        #priority_key="td_error",
        batch_size=32,
    )
"""buffer_lazymemmap = TensorDictReplayBuffer(
    storage=LazyMemmapStorage(size, scratch_dir="/tmp/memmap/"), batch_size=12
)"""
buffer_lazymemmap.extend(data)
print(f"The buffer has {len(buffer_lazymemmap)} elements")
sample = buffer_lazymemmap.sample()
print("sample:", sample['index'])
indices = buffer_lazymemmap.extend(data)
print(indices)











# %%







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



def old_main():

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
