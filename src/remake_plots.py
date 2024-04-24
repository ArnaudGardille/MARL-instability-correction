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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experience-name", type=str, default=None)
    args = parser.parse_args()

    return args
args = parse_args()

path = Path.cwd() / 'results' / args.experience_name #.replace('/', ':')

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


sns.set_theme(style="darkgrid")

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

results_df = pd.read_csv(path / "eval_prio.csv")

for key, values in params_const.items():
    if key in results_df:
        results_df = results_df[results_df[key] == values]

for key, values in params_list_choice.items():
    results_df = results_df[results_df[key].isin(values)]

sns.lineplot(x="Step", y="Average optimality",
             hue=modified_params[0], style=modified_params[1],
             data=results_df, errorbar=('ci', 90))

plt.savefig(path/'eval_prio.png', format='png')
#plt.show()
