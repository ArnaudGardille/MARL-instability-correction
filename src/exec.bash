#!/bin/bash

rbs=('uniform' 'prioritized' 'laber')

for rb in "${rbs[@]}"; do
  for ((run=0; run<=10; run+=1)); do
    /home/nono/.conda/envs/torch_rl/bin/python /home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_new_rb.py --dueling  --rb "$rb" --save-imgs --deterministic-env --run-name "$rb-$run" --seed $run
  done
done
