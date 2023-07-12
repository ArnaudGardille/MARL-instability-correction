#!/bin/bash

#rbs=('uniform' 'prioritized' 'laber')
rbs=('uniform')
#policies=("--boltzmann-policy" "")
adding_eps=("--add-epsilon" "")
correc_loss=("--corrected-loss" "") 
# --load-buffer 

for ((run=0; run<=2; run+=1)); do
  for rb in "${rbs[@]}"; do
    /home/nono/.conda/envs/torch_rl/bin/python "/home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_new_rb_stab.py" --env-normalization  $eps $loss --dueling --rb "$rb"  --run-name "$rb--$eps-$loss-$run-stab-random"  --seed $run --batch-size 1000 --total-timesteps 500000  #--deterministic-env --total-timesteps 10000 --buffer-size 100000
  done
  #for policy in "${policies[@]}"; do
  #  /home/nono/.conda/envs/torch_rl/bin/python "/home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_new_rb_stab.py" --env-normalization  $eps $loss --dueling --rb "$rb"  --run-name "$rb--$eps-$loss-$run-stab-random"  --seed $run --batch-size 10000 #--deterministic-env --total-timesteps 10000 --buffer-size 100000
  #done
  for eps in "${adding_eps[@]}"; do
    /home/nono/.conda/envs/torch_rl/bin/python "/home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_new_rb_stab.py" --env-normalization  $eps $loss --dueling --rb "$rb"  --run-name "$rb--$eps-$loss-$run-stab-random"  --seed $run --batch-size 1000 --total-timesteps 500000  #--deterministic-env --total-timesteps 10000 --buffer-size 100000
  done
  for loss in "${correc_loss[@]}"; do
    /home/nono/.conda/envs/torch_rl/bin/python "/home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_new_rb_stab.py" --env-normalization  $eps $loss --dueling --rb "$rb"  --run-name "$rb--$eps-$loss-$run-stab-random"  --seed $run --batch-size 1000 --total-timesteps 500000  #--deterministic-env --total-timesteps 10000 --buffer-size 100000
  done
done
