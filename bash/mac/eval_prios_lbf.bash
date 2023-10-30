#!/bin/bash

names=('laber_cur_uncorrected' 'likely_cur' 'uniform' )
configs=('--rb laber --prio cur --batch-size 1000' '--rb laber --prio cur --batch-size 1000' '--rb laber --prio cur-past --correct-prio --batch-size 1000' '--rb laber --prio cur --correct-prio --batch-size 1000' '--rb likely --prio cur-past --batch-size 1000' '--rb likely --prio cur --batch-size 1000' '--rb uniform --batch-size 10000')

for ((run=0; run<1; run+=1)); do
  for ((i=0; i<3; i+=1)); do
    /Users/gardille/opt/anaconda3/envs/torch_rl/bin/python /Users/gardille/development/Water-Bomber-Env/src/iql_gym.py  --env-id lbf  --enforce-coop --run-name ${names[i]}-lbf-$run ${configs[i]} --fixed-buffer --load-buffer-from /Users/gardille/development/Water-Bomber-Env/results/lbf-coop-buffer_2023-10-08_17:30:18 --buffer-size 5000000 --train-frequency 1 --total-timesteps 100000 --learning-starts 0 --learning-rate 0.000001 --device cpu --seed $run 
  done
done
