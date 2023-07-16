#!/bin/bash

prios=('td' 'td/past' 'td*cur/past' 'td*cur' 'cur/past' 'cur')

for ((run=0; run<3; run+=1)); do
  for prio in "${prios[@]}"; do
    /home/nono/.conda/envs/torch_rl/bin/python /home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_new_rb_stab.py --env-normalization --load-buffer --dueling  --rb 'laber' --run-name "$prio-$run-small-bs" --seed $run --prio $prio --total-timesteps 500000 --batch-size 500 --learning-starts 0 --corrected-loss
  done
done
