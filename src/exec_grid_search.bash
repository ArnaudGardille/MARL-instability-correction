#!/bin/bash

#rbs=('uniform' 'prioritized' 'laber')
lrs=('0.01' '0.001' '0.0001')

for lr in "${lrs[@]}"; do
  #for ((run=0; run<=10; run+=1)); do
  /home/nono/.conda/envs/torch_rl/bin/python /home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_new_rb_stab.py --dueling  --learning-rate "$lr" --run-name "$lr-grid-random" --deterministic-env --total-timesteps 10000 --buffer-size 100000
  #done
done
