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