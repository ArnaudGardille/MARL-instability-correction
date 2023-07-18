#!/bin/bash

rbs=('uniform' 'prioritized' 'laber')

for rb in "${rbs[@]}"; do
  for ((run=1; run<6; run+=1)); do
    /Users/gardille/opt/anaconda3/envs/torch_rl/bin/python /Users/gardille/development/Water-Bomber-Env/src/iql_new_rb_stab.py  --env-normalization --load-buffer --dueling  --rb "$rb" --run-name "eval-rb-$rb-$run-ser-loss" --seed $run --total-timesteps 100000 --batch-size 1000 --learning-starts 0 --loss-corrected-for-others
    
  done
done
