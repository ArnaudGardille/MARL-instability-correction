#!/bin/bash

rbs=('uniform' 'prioritized' 'laber')

for rb in "${rbs[@]}"; do
  for ((run=0; run<10; run+=1)); do
    /home/nono/.conda/envs/torch_rl/bin/python /home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_new_rb_stab.py --env-normalization --load-buffer --dueling  --rb "$rb" --run-name "$rb-$run-random" --seed $run --total-timesteps 100000 --batch-size 1000 --learning-starts 0 --add-epsilon 
  done
done
