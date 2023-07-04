#!/bin/bash

rbs=('uniform' 'prioritized' 'laber')

for rb in "${rbs[@]}"; do
  for ((run=0; run<=0; run+=1)); do
    /home/nono/.conda/envs/torch_rl/bin/python /home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_new_rb_stab.py --env-normalization --dueling  --rb "$rb" --save-imgs --total-timesteps 500000 --batch-size 10000  --run-name "$rb-$run-random-epsilon" --seed $run --add-epsilon 
  done
done
