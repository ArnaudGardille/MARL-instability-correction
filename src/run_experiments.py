from iql import run_training
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os 
sns.set_theme(style="darkgrid")

from distutils.util import strtobool
import argparse 

#import warnings
#warnings.filterwarnings("ignore")

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-buffer", type=lambda x: bool(strtobool(x)), nargs="?")
    parser.add_argument("--run-name", type=str, default=None)

    # Environment specific arguments
    parser.add_argument("--x-max", type=int)
    parser.add_argument("--y-max", type=int)
    parser.add_argument("--t-max", type=int)
    parser.add_argument("--n-agents", type=int)
    parser.add_argument("--env-normalization", type=lambda x: bool(strtobool(x)), nargs="?")
    parser.add_argument("--num-envs", type=int,
        help="the number of parallel game environments")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str,
        help="the id of the environment")
    parser.add_argument("--load-agents-from", type=str, default=None,
        help="the experiment from which to load agents.")
    parser.add_argument("--load-buffer-from", type=str, default=None,
        help="the experiment from which to load agents.")
    parser.add_argument("--random-policy", type=lambda x: bool(strtobool(x)), nargs="?")
    parser.add_argument("--no-training", type=lambda x: bool(strtobool(x)), nargs="?",
        help="whether to show the video")
    parser.add_argument("--total-timesteps", type=int, nargs="*",
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, nargs="*",
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, nargs="*",)
    parser.add_argument("--tau", type=float, help="the target network update rate")
    parser.add_argument("--evaluation-frequency", type=int, nargs="*",)
    parser.add_argument("--evaluation-episodes", type=int, nargs="*",)
    parser.add_argument("--target-network-frequency", type=int,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int,  nargs="*",
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, nargs="*",
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, nargs="*",
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, nargs="*",
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, nargs="*",
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, nargs="*",
        help="the frequency of training")
    parser.add_argument("--single-agent", type=lambda x: bool(strtobool(x)), nargs="*", 
        help="whether to use a single network for all agents. Identity is the added to observation")
    parser.add_argument("--add-id", type=lambda x: bool(strtobool(x)), nargs="*", 
        help="whether to add agents identity to observation")
    parser.add_argument("--add-epsilon", type=lambda x: bool(strtobool(x)), nargs="*", 
        help="whether to add epsilon to observation")
    parser.add_argument("--dueling", type=lambda x: bool(strtobool(x)), nargs="*", 
        help="whether to use a dueling network architecture.")
    parser.add_argument("--deterministic-env", type=lambda x: bool(strtobool(x)), nargs="*")
    parser.add_argument("--boltzmann-policy", type=lambda x: bool(strtobool(x)), nargs="*")
    parser.add_argument("--loss-corrected-for-others", type=lambda x: bool(strtobool(x)), nargs="*")
    parser.add_argument("--loss-not-corrected-for-prioritized", type=lambda x: bool(strtobool(x)), nargs="*")
    parser.add_argument("--prio", choices=['td_error', 'td-past', 'td-cur-past', 'td-cur', 'cur-past', 'cur'], nargs="*",)
    parser.add_argument("--rb", choices=['uniform', 'prioritized', 'laber'], nargs="*",
        help="whether to use a prioritized replay buffer.")
    args = parser.parse_args()
    # fmt: on
    #assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    return args
args = parse_args()

params = {}
for k, v in vars(args).items():
    if v is not None:
        print(k, ': ', v)
        params[k] = v


import itertools
def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))

param_dicts = list(product_dict(**params))

#param_dicts['total_timesteps'] = 1000
test_params = {
    'rb': 'laber',
    'evaluation_frequency':100,
    'total_timesteps': 1000,
    'evaluation_episodes':100,
}

NB_RUNS = 2

modified_params = [None, None]

for i, (k, v) in enumerate(params.items()):
    if type(v) == list: 
        print(i, k, v)
        modified_params[i] = k

results_df = []
for run in range(NB_RUNS):

    for param_dict in param_dicts:
        run_name= "eval_prio" 

        for k in modified_params:
            if k is not None:
                run_name += "/"+str(k)+':'+str(param_dict[k])
        run_name += '/'+str(run)
        #params['prio'] = prio
        param_dict['total_timesteps'] = 10
        param_dict['evaluation_episodes'] = 2

        steps, avg_opti = run_training(run_name=run_name, seed=run, verbose=False, **param_dict)
        n = len(avg_opti)
        
        results = {
            'Average optimality': avg_opti,
            'Run': [run]*n,
            'Step': steps,
        }
        for k, v in param_dict.items():
            results[k] = [v]*n

        print(results)
        result_df = pd.DataFrame(results)
        print('result_df',result_df)
        results_df.append(result_df)

        print("modified_params: ", modified_params)

experiment_name = ""
for k in modified_params:
    if k is not None:
        experiment_name += str(k)+' '+'-'.join(params[k])+"; "
print("experiment_name:", experiment_name)

results_df = pd.concat(results_df)
path = Path.cwd() / 'results' / experiment_name
os.makedirs(path, exist_ok=True)
print(results_df)
results_df.to_csv(path/ 'eval_prio.csv', index=False)

sns.lineplot(x="Step", y="Average optimality",
             hue=modified_params[0], style=modified_params[1],
             data=results_df)

plt.savefig(path/'eval_prio.svg', format='svg')
#plt.show()
