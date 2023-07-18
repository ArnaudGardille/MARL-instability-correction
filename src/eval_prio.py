from iql import run_training
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os 
sns.set_theme(style="darkgrid")

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