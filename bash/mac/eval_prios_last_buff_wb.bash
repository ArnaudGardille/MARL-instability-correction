#!/bin/bash

#names=('laber_cur-past_uncorrected' 'laber_cur_uncorrected' 'laber_cur-past' 'laber_cur' 'likely_cur-past' 'likely_cur' 'uniform' )
#configs=('--rb laber --prio cur-past --batch-size 1000' '--rb laber --prio cur --batch-size 1000' '--rb laber --prio cur-past --correct-prio --batch-size 1000' '--rb laber --prio cur --correct-prio --batch-size 1000' '--rb likely --prio cur-past --batch-size 1000' '--rb likely --prio cur --batch-size 1000' '--rb uniform --batch-size 10000')

names=('laber_cur_uncorrected' 'laber_cur-past_uncorrected' 'likely_cur' 'likely_cur-past' 'uniform' )
configs=('--rb laber --prio cur --batch-size 1000' '--rb laber --prio cur-past --batch-size 1000'  '--rb likely --prio cur --batch-size 1000'  '--rb likely --prio cur-past --batch-size 1000' '--rb uniform --batch-size 10000')
lr=(0.01 0.001 0.0001 0.00001)



for ((run=1; run<10; run+=1)); do
  for ((i=0; i<5; i+=1)); do
    for ((j=0; j<4; j+=1)); do
      /Users/gardille/opt/anaconda3/envs/torch_rl/bin/python /Users/gardille/development/Water-Bomber-Env/src/iql_gym.py  --env-id water-bomber  --enforce-coop --run-name ${names[i]}-${lr[j]}-last-buff-$run ${configs[i]} --fixed-buffer --load-buffer-from /Users/gardille/development/Water-Bomber-Env/results/water-bomber-buffer --buffer-size 200000 --train-frequency 1 --total-timesteps 10000 --learning-starts 0 --learning-rate ${lr[j]} --device cpu --seed $run  --last-buffer
    done
  done
done