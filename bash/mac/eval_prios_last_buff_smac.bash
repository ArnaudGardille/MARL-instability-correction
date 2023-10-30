#!/bin/bash

names=('laber_cur_uncorrected' 'laber_cur-past_uncorrected' 'likely_cur' 'likely_cur-past' 'uniform' )
configs=('--rb laber --prio cur --batch-size 1000 --correct-prio False' '--rb laber --prio cur-past --batch-size 1000 --correct-prio False'  '--rb likely --prio cur --batch-size 1000'  '--rb likely --prio cur-past --batch-size 1000' '--rb uniform --batch-size 10000')
lr=(0.01 0.001 0.0001 0.00001)



for ((run=1; run<3; run+=1)); do
  for ((i=0; i<2; i+=1)); do
    for ((j=3; j<4; j+=1)); do
      /Users/gardille/opt/anaconda3/envs/torch_rl/bin/python /Users/gardille/development/Water-Bomber-Env/src/iql_gym.py  --env-id smac  --enforce-coop --run-name ${names[i]}-${lr[j]}-smac-last_buff-$run ${configs[i]} --fixed-buffer --load-buffer-from /Users/gardille/development/Water-Bomber-Env/results/full_medium_smac --buffer-size 2000000 --train-frequency 1 --total-timesteps 100000 --learning-starts 0 --learning-rate ${lr[j]} --device cpu --seed $run   --last-buffer
    done
  done
done