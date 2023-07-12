#!/bin/bash

#rbs=('uniform' 'prioritized' 'laber')
lrs=('0.01' '0.001' '0.0001')
#lrs=('0.5' '0.1' '0.05')

for lr in "${lrs[@]}"; do
  #for ((run=0; run<=10; run+=1)); do
  /home/nono/.conda/envs/torch_rl/bin/python "/home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_new_rb_stab.py" --dueling --rb prioritized --total-timesteps 100000 --train-frequency 10 --load-buffer --batch-size 10000 --learning-rate "$lr" --learning-starts 10000  --add-epsilon --run-name "$lr-grid-random"  
  #done
done
