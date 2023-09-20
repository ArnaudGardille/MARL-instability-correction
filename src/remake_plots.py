#NB_RUNS = 3

#from iql import run_training
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


#import warnings
#warnings.filterwarnings("ignore")

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--x", type=float, default=11.7)
    parser.add_argument("--y", type=float, default=8.27)
    args = parser.parse_args()
    # fmt: on
    #assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    return args
args = parse_args()

#name = args.name.replace('/', ':')
name = args.name
path = Path.cwd() / 'results' / name

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


sns.set_theme(style="darkgrid")
#sns.set(rc={'figure.figsize':(args.x, args.y)})

with open(path/'params_const.yaml', 'r') as f:
    params_const = yaml.safe_load(f)

with open(path/'params_list_choice.yaml', 'r') as f:
    params_list_choice = yaml.safe_load(f)


print("params_list_choice:", params_list_choice)
print("params_const:", params_const)

import itertools
def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))

params_list_choices_dicts = list(product_dict(**params_list_choice))


modified_params = [None, None]

i = 0
for k, v in params_list_choice.items():
    if type(v) == list: 
        print(i, k, v)
        modified_params[i] = k
        i += 1

#print("modified_params:", modified_params)




results_df = pd.read_csv(path / "eval_prio.csv")

for key, values in params_const.items():
    if key in results_df:
        results_df = results_df[results_df[key] == values]

for key, values in params_list_choice.items():
    results_df = results_df[results_df[key].isin(values)]

#print(results_df)

sns.lineplot(x="Step", y="Average optimality",
             hue=modified_params[0], style=modified_params[1],
             data=results_df, errorbar=('ci', 90))

plt.savefig(path/'eval_prio.png', format='png')
#plt.show()
