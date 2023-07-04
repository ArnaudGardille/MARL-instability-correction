#!/bin/bash

#rbs=('uniform' 'prioritized' 'laber')
rbs=('uniform')
policies=("--boltzmann-policy" "")
adding_eps=("--add-epsilon" "")
correc_loss=("--corrected-loss" "") 

for ((run=0; run<=5; run+=1)); do
  for loss in "${correc_loss[@]}"; do
    /home/nono/.conda/envs/torch_rl/bin/python "/home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_new_rb_stab.py" --env-normalization $loss --dueling --save-imgs --run-name " ---$loss-$run-stab-random"  --seed $run --batch-size 10000 #--deterministic-env --total-timesteps 10000 --buffer-size 100000
  done
done
