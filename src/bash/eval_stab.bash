#!/bin/bash

#rbs=('uniform' 'prioritized' 'laber')
#policies=("--boltzmann-policy" "")
stabs=("--corrected-loss" "--add-epsilon" "")
# --load-buffer 

for ((run=0; run<=2; run+=1)); do
  for stab in "${stabs[@]}"; do
    /home/nono/.conda/envs/torch_rl/bin/python "/home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_new_rb_stab.py" --env-normalization --load-buffer --dueling --run-name "$stab-$run"  --seed $run --batch-size 1000 --total-timesteps 200000  #--deterministic-env --total-timesteps 10000 --buffer-size 100000
  done
done
