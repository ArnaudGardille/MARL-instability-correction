from iql import run_training
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os 
sns.set_theme(style="darkgrid")

from distutils.util import strtobool
import argparse 

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-buffer", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
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
    parser.add_argument("--prio", choices=['td_error', 'td-past', 'td-cur-past', 'td-cur', 'cur-past', 'cur'], default='td_error')
    parser.add_argument("--rb", choices=['uniform', 'prioritized', 'laber'], default='uniform',
        help="whether to use a prioritized replay buffer.")
    args = parser.parse_args()
    # fmt: on
    #assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    return args


prios=('td_error', 'td-cur', 'cur', 'td-past', 'td-cur-past', 'cur-past' )
# 'td/past' 'td*cur/past' 'cur/past' 

params = {
    'rb': 'laber',
    'evaluation_frequency':100,
    'total_timesteps': 1000,
    'evaluation_episodes':100,
}

NB_RUNS = 3

results = {
    'Average optimality': [],
    'Run': [],
    'Prio': [],
    'Step': [],
}

results_df = []
for run in range(NB_RUNS):

    for prio in prios:
        run_name= "eval_prio/{prio}/{run}"
        params['prio'] = prio

        avg_opti = run_training(run_name=run_name, seed=run, **params)
        n = len(avg_opti)
        
        results = {
            'Average optimality': avg_opti,
            'Run': [run]*n,
            'Prio': [prio]*n,
            'Step': [i for i in range(0, params['total_timesteps'], params['evaluation_frequency'])],
        }

        print(results)
        result_df = pd.DataFrame(results)
        print('result_df',result_df)
        results_df.append(result_df)

results_df = pd.concat(results_df)
path = Path.cwd() / 'results' 
os.makedirs(path, exist_ok=True)
print(results_df)
results_df.to_csv(path/ 'eval_prio.csv', index=False)

sns.lineplot(x="Step", y="Average optimality",
             hue="Prio", #style="event",
             data=results_df)

plt.savefig(path/'eval_prio.svg', format='svg')
plt.show()