#!/bin/bash

names=('laber_cur-past_uncorrected' 'laber_cur_uncorrected' 'laber_cur-past' 'laber_cur' 'likely_cur-past' 'likely_cur' 'uniform' )
configs=('--rb laber --prio cur-past --batch-size 1000' '--rb laber --prio cur --batch-size 1000' '--rb laber --prio cur-past --correct-prio --batch-size 1000' '--rb laber --prio cur --correct-prio --batch-size 1000' '--rb likely --prio cur-past --batch-size 1000' '--rb likely --prio cur --batch-size 1000' '--rb uniform --batch-size 10000')


for ((run=0; run<1; run+=1)); do
  for ((i=0; i<7; i+=1)); do
    /home/nono/.conda/envs/torch_rl/bin/python /home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_gym.py --env-id lbf --run-name ${names[i]}-lbf-last_buff-$run ${configs[i]} --fixed-buffer --load-buffer-from /home/nono/Documents/Dassault/Water-Bomber-Env/results/full_medium_lbf_2023-10-04_10:25:28 --buffer-size 5000000 --train-frequency 1 --total-timesteps 100000 --learning-starts 0 --learning-rate 0.001 --device cpu --seed $i --last-buffer
  done
done
