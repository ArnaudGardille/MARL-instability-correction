#!/bin/bash

#rbs=('uniform' 'prioritized' 'laber')
rbs=('uniform')
policies=("--boltzmann-policy" "")
adding_eps=("--add-epsilon" "")
correc_loss=("--corrected-loss" "") 

for ((run=0; run<=5; run+=1)); do
  for rb in "${rbs[@]}"; do
    /home/nono/.conda/envs/torch_rl/bin/python "/home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_new_rb_stab.py" --env-normalization $policy $eps $loss --dueling --rb "$rb" --save-imgs --run-name "$rb-$policy-$eps-$loss-$run-stab-random"  --seed $run --batch-size 10000 #--deterministic-env --total-timesteps 10000 --buffer-size 100000
  done
  for policy in "${policies[@]}"; do
    /home/nono/.conda/envs/torch_rl/bin/python "/home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_new_rb_stab.py" --env-normalization $policy $eps $loss --dueling --rb "$rb" --save-imgs --run-name "$rb-$policy-$eps-$loss-$run-stab-random"  --seed $run --batch-size 10000 #--deterministic-env --total-timesteps 10000 --buffer-size 100000
  done
  for eps in "${adding_eps[@]}"; do
    /home/nono/.conda/envs/torch_rl/bin/python "/home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_new_rb_stab.py" --env-normalization $policy $eps $loss --dueling --rb "$rb" --save-imgs --run-name "$rb-$policy-$eps-$loss-$run-stab-random"  --seed $run --batch-size 10000 #--deterministic-env --total-timesteps 10000 --buffer-size 100000
  done
  for loss in "${correc_loss[@]}"; do
    /home/nono/.conda/envs/torch_rl/bin/python "/home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_new_rb_stab.py" --env-normalization $policy $eps $loss --dueling --rb "$rb" --save-imgs --run-name "$rb-$policy-$eps-$loss-$run-stab-random"  --seed $run --batch-size 10000 #--deterministic-env --total-timesteps 10000 --buffer-size 100000
  done
done
