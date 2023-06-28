#!/bin/bash

#rbs=('uniform' 'prioritized' 'laber')
rbs=('uniform')
policies=("" "--boltzmann-policy")
adding_eps=("" "--add-epsilon")
correc_loss=("" "--corrected-loss")

for rb in "${rbs[@]}"; do
  for policy in "${policies[@]}"; do
    for eps in "${adding_eps[@]}"; do
      for loss in "${correc_loss[@]}"; do
        for ((run=0; run<=0; run+=1)); do
          /home/nono/.conda/envs/torch_rl/bin/python "/home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_new_rb_stab.py" $policy $eps $loss --dueling --rb "$rb" --save-imgs --run-name "$rb-$policy-$eps-$loss-stab-random"  --seed $run --deterministic-env --total-timesteps 10000 --buffer-size 100000
        done
      done
    done
  done
done
