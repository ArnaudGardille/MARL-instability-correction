#!/bin/bash

rbs=('uniform' 'prioritized' 'laber')

for rb in "${rbs[@]}"; do
  for ((run=0; run<=10; run+=1)); do
    /home/nono/.conda/envs/torch_rl/bin/python /home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_new_rb_stab.py --env-normalization --dueling  --rb "$rb" --save-imgs --run-name "$rb-$run" --seed $run --total-timesteps 10000 --deterministic-env 
  done
done
