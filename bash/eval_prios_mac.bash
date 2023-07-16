#!/bin/bash

prios=('td' 'td*cur' 'cur')
# 'td/past' 'td*cur/past' 'cur/past' 

for ((run=0; run<3; run+=1)); do
  for prio in "${prios[@]}"; do
    /Users/gardille/opt/anaconda3/envs/torch_rl/bin/python /Users/gardille/development/Water-Bomber-Env/src/iql_new_rb_stab.py --env-normalization --load-buffer --dueling  --rb 'laber' --run-name "$prio-$run-random" --seed $run --prio $prio --total-timesteps 200000 --batch-size 1000 --learning-starts 0 --corrected-loss --n-agents=3
  done
done
