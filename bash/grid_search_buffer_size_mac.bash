#!/bin/bash

#rbs=('uniform' 'prioritized' 'laber')
sizes=(1000 10000 100000 1000000)
#lrs=('0.5' '0.1' '0.05')

for size in "${sizes[@]}"; do
  #for ((run=0; run<=10; run+=1)); do
  /Users/gardille/opt/anaconda3/envs/torch_rl/bin/python /Users/gardille/development/Water-Bomber-Env/src/iql_new_rb_stab.py  --env-normalization --dueling  --total-timesteps 100000 --batch-size $size --run-name "buffer-size-$size"  
  #done
done
