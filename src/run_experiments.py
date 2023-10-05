from iql_gym import run_training
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from pathlib import Path
import os 
import datetime
from distutils.util import strtobool
import argparse 
import yaml

# Warnings supression
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("error", category=RuntimeWarning)

sns.set_theme(style="darkgrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-buffer", type=lambda x: bool(strtobool(x)) , const=True, nargs="?")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, choices=['cpu', 'mps', 'cuda'], nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--nb-runs", type=int, default=3,
        help="the number of trainings that will be averaged")

    # Environment specific arguments
    parser.add_argument("--x-max", type=int)
    parser.add_argument("--y-max", type=int)
    parser.add_argument("--t-max", type=int)
    parser.add_argument("--n-agents", type=int, nargs="*")
    parser.add_argument("--env-normalization", type=lambda x: bool(strtobool(x)), nargs="*")
    parser.add_argument("--env-id", choices=['simultaneous', 'water-bomber', 'smac', 'lbf', 'mpe'] ,default='simultaneous',
        help="the id of the environment")
    parser.add_argument("--map", type=str, nargs="*" #choices=['10m_vs_11m', '27m_vs_30m', '2c_vs_64zg', '2s3z', '2s_vs_1sc', '3s5z', '3s5z_vs_3s6z', '3s_vs_5z', 'bane_vs_bane', 'corridor', 'MMM', 'MMM2']
        ,help="Select the map when using SMAC")
    parser.add_argument("--enforce-coop", type=lambda x: bool(strtobool(x)), nargs="*",
        help="Coop version for lbf, and mix up rewards for simultaneous")
    # Algorithm specific arguments
    parser.add_argument("--fixed-buffer", type=lambda x: bool(strtobool(x)), nargs="?", const=True,
        help="Nothing will be added to the buffer(only useful if it has been loaded)")
    parser.add_argument("--load-agents-from", type=str, default=None,
        help="the experiment from which to load agents.")
    parser.add_argument("--load-buffer-from", type=str, default=None,
        help="the experiment from which to load agents.")
    parser.add_argument("--random-policy", type=lambda x: bool(strtobool(x)) , const=True, nargs="?")
    parser.add_argument("--no-training", type=lambda x: bool(strtobool(x)) , const=True, nargs="?",
        help="The agents won't be trained")
    parser.add_argument("--total-timesteps", type=int, nargs="*",
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, nargs="*",
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, nargs="*",
        help="the replay memory buffer size")
    parser.add_argument("--buffer-on-disk", type=lambda x: bool(strtobool(x)), nargs="?", const=True,
        help="The buffer will be stored on disk. Useful if it is to big to fit in the RAM.")
    parser.add_argument("--gamma", type=float, nargs="*",
        help="the discount factor")
    parser.add_argument("--tau", type=float, help="the target network update rate")
    parser.add_argument("--evaluation-frequency", type=int, nargs="*",
        help="number of episodes between two evaluations")
    parser.add_argument("--evaluation-episodes", type=int, nargs="*",
        help="number of evaluation episodes that will be averaged")
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
    parser.add_argument("--add-id", type=lambda x: bool(strtobool(x)) , const=True, nargs="?", 
        help="whether to add a one-hot envodding of agents identity to observation")
    parser.add_argument("--add-epsilon", type=lambda x: bool(strtobool(x)) , nargs="*", help="whether to add epsilon to observation")
    parser.add_argument("--add-others-explo", type=lambda x: bool(strtobool(x)), nargs="*"
        ,help="whether to add a boolean vector of wheter each other agent is exploring to observation")
    parser.add_argument("--dueling", type=lambda x: bool(strtobool(x)), nargs="*", 
        help="whether to use a dueling network architecture.")
    parser.add_argument("--deterministic-env", type=lambda x: bool(strtobool(x)) , const=True, nargs="?")
    parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)) , const=True, nargs="?")
    parser.add_argument("--loss-correction-for-others", choices=['none', 'td_error', 'td-past', 'td-cur-past', 'td-cur', 'cur-past', 'cur'], nargs="*"
        ,help="How to correct the loss for the others evolution. please refer to the article for more explanations")
    parser.add_argument("--correction-modification", choices=['none', 'sqrt', 'sigmoid', 'normalize'] , nargs="*"
        ,help="function that will be applied to the loss correction")
    parser.add_argument("--clip-correction-after", type=float, nargs="*"
        ,help="Allows to clip the correction")
    parser.add_argument("--prioritize-big-buffer", type=lambda x: bool(strtobool(x)), nargs="*"
        ,help="Wheter to do prioritized experience replay on the big replay buffer (when using likely of laber)")
    parser.add_argument("--prio", choices=['none','td_error', 'td-past', 'td-cur-past', 'td-cur', 'cur-past', 'cur', 'past'], nargs="*"
        ,help="Select the priorisation quantity for PER of Laber")
    parser.add_argument("--rb", choices=['uniform', 'prioritized', 'laber', 'likely', 'correction'], nargs="*",
        help="whether to use a uniform, prioritized, Laber of Likely replay buffer.")
    parser.add_argument("--filter", choices=['none', 'td_error', 'td-past', 'td-cur-past', 'td-cur', 'cur-past', 'cur', 'past'], nargs="*"
        ,help="Select the priorisation quantity for Likely")
    parser.add_argument("--correct-prio", type=lambda x: bool(strtobool(x)), default=True, nargs="*")
    
    args = parser.parse_args()

    return args
args = parse_args()

params_list_choice = {}
params_const = {}

for k, v in vars(args).items():
    if v is not None:
        print(k, ': ', v)
        if type(v)==list:
            if len(v)>1:
                params_list_choice[k] = v
            else:
                params_const[k] = v[0]
        else:
            params_const[k] = v

if 'prio' in params_list_choice:
    print("Switching to a Laber priorisation")
    params_const['rb'] = 'laber'

print("params_list_choice:", params_list_choice)
print("params_const:", params_const)

import itertools
def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))

params_list_choices_dicts = list(product_dict(**params_list_choice))

test_params = {
    'rb': 'laber',
    'evaluation_frequency':100,
    'total_timesteps': 1000,
    'evaluation_episodes':100,
}

modified_params = [None, None]

i = 0
for k, v in params_list_choice.items():
    if type(v) == list: 
        print(i, k, v)
        modified_params[i] = k
        i += 1

experiment_name= '{date:%Y-%m-%d_%H:%M:%S}'.format( date=datetime.datetime.now() ) 
for k in params_const:
    experiment_name += '-' + str(k)
for k in params_list_choice:
    experiment_name += '-' + str(k)
print("experiment_name:", experiment_name)

path = Path.cwd() / 'results' / experiment_name
os.makedirs(path, exist_ok=True)

results_df = []
for run in range(args.nb_runs):
    print("Run", run)

    for params_choice in params_list_choices_dicts:
        run_name= "" 
        for k in params_const:
            run_name += str(k)+':'+str(params_const[k]) + "/"
        for k in params_choice:
            run_name += str(k)+':'+str(params_choice[k]) + "/"
        run_name += str(run)
        print("Run name:", run_name)

        param_dict = {**params_choice, **params_const}
        
        
        steps, avg_opti = run_training(path=path, run_name=run_name, seed=run, **param_dict)
        n = len(avg_opti)
        
        results = {
            'Average optimality': avg_opti,
            'Run': [run]*n,
            'Step': steps,
        }
        for k, v in params_choice.items():
            results[k] = [v]*n

        result_df = pd.DataFrame(results)
        results_df.append(result_df)

with open(path/'params_const.yaml', 'w') as f:
    yaml.dump(params_const, f, default_flow_style=False)

with open(path/'params_list_choice.yaml', 'w') as f:
    yaml.dump(params_list_choice, f, default_flow_style=False)

results_df = pd.concat(results_df)
results_df.to_csv(path/ 'eval_prio.csv', index=False)

sns.lineplot(x="Step", y="Average optimality",
             hue=modified_params[0], style=modified_params[1],
             data=results_df, errorbar=('ci', 90))

plt.savefig(path/'eval_prio.png', format='png')
#plt.show()
