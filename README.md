# Addressing Perceived Environment Instability

This repository implements Independent Q-Learning and allows you to experiment with various solutions to address perceived environment instability. This problem arises due to other agents' learning, causing their actions stored in the replay buffer to no longer correspond to their current policies.

Here are the implemented solutions:

* Adding epsilon to the state
* Adding others' exploration noise to the state
* Correcting the loss function
* Likely Transition Selection Algorithm

## Features

* Fully configure the IQL algorithm through YAML parameter files and command lines.
* Average multiple training runs, run experiments with different parameters, and plot the results.
* Save and load both agents' networks and the replay buffer.
* Easily add your own multi-agent gym environments.

## Supported Environments:

* 'simultaneous': The Simultaneous-attack environment
* 'water-bomber': The Water-bomber environment
* 'smac': The Starcraft micromanagement environment

## Installation

```
conda create -n dassault-env python=3.9 -y
conda activate dassault-env
pip3 install torch torchvision torchaudio
pip install -r src/requirements.txt

If you have troubles with requirements.txt, you can try requirements_specific.txt where versions are specified.

Install the simultaneous and water-bomber environments from https://github.com/ArnaudGardille/my-envs

Install the smac environment from https://github.com/ArnaudGardille/smaclite
```

## Training:

Default parameters are loaded from `config/MAP_NAME/default.yaml`. You can modify them for a specific run through the command line.

Run a single training with:

```
python src/iql_gym.py --env-id simultaneous --n-agents 8 --run-name test
```

Access the parameters and their explanations with:

```
python src/iql_gym.py -h
```

Visualize the training by passing the experiment's directory to TensorBoard:

```
tensorboard --logdir=.../test
```

You can also average over several training runs, run experiments with different parameters, and plot the results:

```
python src/run_experiments.py -env-id simultaneous --n-agents 2 4 6 8 10 --nb-runs 10
```

If you want to modify things in the experiment folder and remake the plots, use:

```
python src/remake_plots.py --experience-name test
```
