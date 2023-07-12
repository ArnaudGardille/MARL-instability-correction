#!/bin/bash

prios=('td' 'td/past' 'td*cur/past' 'td*cur' 'cur/past' 'cur')

for ((run=0; run<10; run+=1)); do
  for prio in "${prios[@]}"; do
    /home/nono/.conda/envs/torch_rl/bin/python /home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_new_rb_stab.py --env-normalization --load-buffer --dueling  --rb 'laber' --run-name "$prio-$run-random" --seed $run --prio $prio --total-timesteps 100000 --batch-size 1000 --learning-starts 0 --add-epsilon 
  done
done
