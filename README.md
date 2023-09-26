# Facing the percieved environment instability

This repo implements Independant-Q-Learning, and offers the possibility to try various solution to fact the percieved environment instability. This probalem is due to others learning,  which causes their actions we store in the replay buffer not to correspound to others policies anymore.

The encoded solutions are:
- adding epsilon to the state
- adding others exploration to the state
- correcting the loss function
- the Likely Transition Selection Algorithm


## Features
- The IQL algorithm is fully configurable through yaml parameter files and command lines
- Possibility to average several training, to run several experiments with different parameters and to plot the results
- Possibility to save and load both agents networks and replay buffer
- Easy to add your own multi-agents gym environements

## Currently available environments 
- 'simultaneous'  : The Simultaneous-attack environment
- 'water-bomber'  : The Water-bomber environment
- 'smac'          : The Starcraft micro-managements environment

## Installation

```
conda create -n dassault-env python=3.9  -y
conda activate dassault-env
pip3 install torch torchvision torchaudio
pip install -r src/requirements.txt
```

If you have troubles with requirements.txt, you can try requirements_specific.txt where versions are specified.

install the simultaneous and water-bomber environements from https://github.com/ArnaudGardille/my-envs

install the smac environement from https://github.com/ArnaudGardille/smaclite


## Running

The delaut parameters are loaded from config/MAP_NAME/default.yaml
You can modify them for a specific run through command line. 


You can run a single training with
```
python src/iql_gym.py --env-id simultaneous --n-agents 8 --run-name test 
```
you can have access to the parameters and their explanations with -h:
```
python src/iql_gym.py -h
```
And visualise the tranining by passing the experiment's directory to tensorboard
```
tensorboard --logdir=.../test
```

You can also average over several training, run experiments with different parameters and plot the results
```
python src/run_experiments.py  -env-id simultaneous --n-agents 2 4 6 8 10 --nb-runs 10
```
 
If you want to modify things in the explerience forlder and re-make the plots, use:
```
python src/remake_plots.py --experience-name test                 
```




python src/run_experiments.py  --env-id  lbf --enforce-coop True False  --rb uniform laber likely --prio cur-past --add-epsilon --add-others-explo

 --env-id  lbf --enforce-coop True False --save-buffer --s
ython src/run_experiments.py  --env-id  lbf --enforce-coop True False -
-save-buffer --save-model --rb uniform laber likely --prio cur-past
  --rb  

  python src/run_experiments.py --rb uniform prioritized laber likely correction --nb-runs 3 --env-id water-bomber 
 python src/run_experiments.py  --rb prioritized likely --filter td-cur-past --nb-runs 10 --env-id water-bomber

python src/run_experiments.py --add-epsilon True False --add-others-explo True False --nb-runs 10 --env-id water-bomber 
python src/run_experiments.py --rb uniform likely --prio none --filter cur-past cur --nb-runs 3 --env-id water-bomber
python src/run_experiments.py  --rb prioritized likely --filter td-cur --nb-runs 10 --env-id water-bomber


python src/run_experiments.py --rb laber --prio none cur cur-past

python src/run_experiments.py --rb uniform laber likely --prio cur --filter cur-past prioritize-big-buffer False 

--loss-correction-for-others cur
python src/run_experiments.py --rb laber --prio none cur cur-past --loss-correction-for-others none cur cur-past  

python src/run_experiments.py --rb laber --prio none cur cur-past --loss-correction-for-others none cur cur-past  --nb-runs 100


python src/run_experiments.py --rb uniform laber likely --prio cur-past --filter cur-past --prioritize-big-buffer False --nb-runs 100

python src/remake_plots.py --name                   

python src/run_experiments.py --rb laber likely --prio cur-past  --filter cur-past --loss-correction-for-others none cur cur-past  

python src/run_experiments.py --rb laber --prio cur-past td_error td-cur-past --prioritize_big_buffer False  

python src/run_experiments.py --rb uniform laber likely --prio cur-past --filter cur-past

--loss_correction_for_others none cur cur-past


python src/run_experiments.py --rb uniform laber likely --prio cur-past --filter cur-past --env-id smac --single-agent True
--prioritize-big-buffer True

/home/nono/.conda/envs/torch_rl/bin/python /home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_gym.py --total-timesteps 1000000
 --buffer-size 20000000


 python src/run_experiments.py --rb laber likely --prio cur-past  --filter  none cur past cur-past --loss-correction-for-others none cur cur-past  
--map 10m_vs_11m 27m_vs_30m 2c_vs_64zg 2s3z 2s_vs_1sc 3s5z 3s5z_vs_3s6z 3s_vs_5z bane_vs_bane corridor mmm mmm2

python src/run_experiments.py --env-id smac --map 10m_vs_11m 27m_vs_30m 2c_vs_64zg 2s3z 2s_vs_1sc 3s5z 3s5z_vs_3s6z 3s_vs_5z bane_vs_bane corridor mmm mmm2 --nb-runs 1 --device cpu


python src/run_experiments.py --rb laber --prio none cur past --loss-correction-for-others none cur cur-past 

python src/run_experiments.py --rb laber --prio none cur past --loss-correction-for-others none cur cur-past --nb-runs 3 --env-id water-bomber
python src/run_experiments.py --rb laber --prio none cur past --loss-correction-for-others none cur cur-past  --nb-runs 100

/home/nono/.conda/envs/torch_rl/bin/python /home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_gym.py --env-id smac --use-state

python src/run_experiments.py --rb uniform laber likely --prio cur-past --filter cur-past --env-id smac --nb-runs 2
python src/run_experiments.py --add-epsilon True False  --env-id smac --nb-runs 3
python src/run_experiments.py --add-others-explo True False --env-id smac --nb-runs 3
python src/run_experiments.py --rb uniform prioritized laber --env-id smac --nb-runs 2

/home/nono/.conda/envs/torch_rl/bin/python /home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_gym.py --single-agent --add-id --env-id smac --map MMM2 --save-model --save-buffer --use-state

python src/run_experiments.py --rb uniform laber likely --prio cur-past  --filter cur-past  --nb-runs 10 --env-id water-bomber


 python src/run_experiments.py --rb uniform laber likely --prio td-cur-past  --filter td-cur-past --loss-correction-for-others cur-past --correction-modification sqrt --nb-runs 3 --env-id water-bomber
 python src/run_experiments.py --rb uniform laber likely --prio td-cur-past  --filter td-cur-past --loss-correction-for-others cur-past  --correction-modification sqrt --nb-runs 100
 --prioritize-big-buffer
--correction-modification sqrt

 python src/run_experiments.py --rb laber --filter  cur-past --loss-correction-for-others cur-past  --prioritize-big-buffer True --nb-runs 3 --env-id water-bomber

 python src/run_experiments.py --rb laber --filter  cur-past --loss-correction-for-others cur-past  --prioritize-big-buffer True  --nb-runs 100

 python src/run_experiments.py --loss-correction-for-others cur-past --correction-modification none sqrt sigmoid normalize  --nb-runs 100

  python src/run_experiments.py --buffer-size 100 1000 10000  --nb-runs 100

  python src/run_experiments.py --rb laber likely --prio cur-past 
--filter  cur-past --loss-correction-for-others cur-past  --prioritize-big-buffer True  --nb-runs 100 --correction-modification none sqrt

  python src/run_experiments.py --rb laber --prio td-cur-past --loss-correction-for-others cur-past  --prioritize-big-buffer False   --correction-modification none sqrt --nb-runs 100

  python src/run_experiments.py --rb likely --prio cur-past --filter  cur-past --loss-correction-for-others cur-past  --prioritize-big-buffer True  --nb-runs 100 --correction-modification none sqrt


   python src/run_experiments.py --rb laber --prio td-cur-past --loss-correction-for-others cur-past  --prioritize-big-buffer False   --correction-modification none sqrt --nb-runs 100

   python src/run_experiments.py --rb laber --prio td-cur-past --loss-correction-for-others cur-past  --prioritize-big-buffer False   --correction-modification none sqrt --nb-runs 3 --env-id water-bomber


   python src/run_experiments.py --rb uniform laber likely --prio cur-past  --filter cur-past --loss-correction-for-others none cur cur-past  


python src/run_experiments.py --rb laber likely --prio td-cur-past  --filter cur-past --prioritize-big-buffer True   


python src/run_experiments.py  --buffer-size 5000 7500 --nb-runs 100

python src/run_experiments.py --rb likely --prio td-cur-past  --filter cur-past --prioritize-big-buffer True --nb-runs 100 
--nb-runs 3 --env-id water-bomber


 python src/run_experiments.py --rb uniform prioritized laber --prio td_error 

 python src/run_experiments.py --batch-size 

python src/iql_gym.py --save-model --env-id smac --n-agents 5 --buffer-size 10000000 --total-timesteps 100000 --device cuda
python iql_gym.py --save-model --env-id smac --n-agents 5 --total-timesteps 100000
python src/iql_gym.py --save-model --env-id smac --n-agents 5 --total-timesteps 100000 --buffer-size 10000000

python src/iql_gym.py --env-id smac --n-agents 5 --no-training --load-agents-from test --visualisation

python src/iql_gym.py  --save-model --env-id smac --total-timesteps 100000 --map MMM2 --buffer-size 10000000

python src/iql_gym.py  --total-timesteps 100000 --save-model --env-id smac --buffer-size 10000000 --map MMM2

--map 27m_vs_30m --single-agent

python src/iql_gym.py --save-model --env-id smac --total-timesteps 10000 --buffer-size 1000000 --map 10m_vs_11m --single-agent  --n-agents 10 --load-agents-from 2023-09-18_02:00:55

python src/run_experiments.py --rb uniform laber --batch-size 32 128

python src/iql_gym.py --save-model --env-id smac --n-agents 5 --total-timesteps 100000 --buffer-size 15000000 --save-buffer

python src/iql_gym.py  --total-timesteps 100000 --save-model --env-id smac --buffer-size 15000000 --run-name big-rb --save-buffer --buffer-on-disk

python src/iql_gym.py  --total-timesteps 100000 --save-model --env-id smac --buffer-size 1500000 --run-name medium-rb --save-buffer

python src/iql_gym.py  --total-timesteps 100000 --save-model --env-id smac --buffer-size 1500000 --run-name medium-rb_epsilon --save-buffer --add-epsilon

python src/iql_gym.py  --total-timesteps 100000 --save-model --env-id smac --buffer-size 1500000 --run-name medium-rb_explo --save-buffer --add-others-explo

python src/iql_gym.py  --total-timesteps 100000 --save-model --env-id smac --buffer-size 1500000 --run-name medium-solo-10 --save-buffer --n-agents 10 --single-agent --map 10m_vs_11m


python src/iql_gym.py  --total-timesteps 100000 --save-model --env-id smac --buffer-size 15000000 --run-name big-rb --save-buffer --buffer-on-disk

python src/iql_gym.py  --total-timesteps 100000 --save-model --env-id smac --buffer-size 1500000 --run-name medium-rb --save-buffer

python src/iql_gym.py  --total-timesteps 100000 --save-model --env-id smac --buffer-size 1500000 --run-name medium-rb_epsilon --save-buffer --add-epsilon

python src/iql_gym.py  --total-timesteps 100000 --save-model --env-id smac --buffer-size 1500000 --run-name medium-rb_explo --save-buffer --add-others-explo

python src/iql_gym.py  --total-timesteps 100000 --save-model --env-id smac --buffer-size 1500000 --run-name medium-solo-10 --save-buffer --n-agents 10 --single-agent --map 10m_vs_11m

python src/iql_gym.py --env-id lbf  --enforce-coop --save-buffer --save-model --run-name lbf-coop-buffer
python src/iql_gym.py --env-id lbf  --enforce-coop --save-buffer --save-model --run-name lbf-coop-augm-buffer --add-epsilon --add-others-explo
python src/iql_gym.py --env-id lbf  --enforce-coop --save-buffer --save-model --run-name lbf-single-buffer --single-agent --add-id
python src/iql_gym.py --env-id lbf --save-buffer --save-model --run-name lbf-buffer.

python src/iql_gym.py --env-id smac --save-buffer --save-model --run-name smac-small-buffer

python /home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_gym.py --env-id lbf --save-buffer --save-model --use-state

python /home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_gym.py --env-id smac --save-buffer --save-model --use-state --buffer-size 1000000

