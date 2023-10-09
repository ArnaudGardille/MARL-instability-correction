#!/bin/bash

names=('laber_cur-past_uncorrected' 'laber_cur_uncorrected' 'laber_cur-past' 'laber_cur' 'likely_cur-past' 'likely_cur' 'uniform' )
configs=('--rb laber --prio cur-past' '--rb laber --prio cur' '--rb laber --prio cur-past --correct-prio' '--rb laber --prio cur --correct-prio' '--rb likely --prio cur-past' '--rb uniform --batch-size 10000')


for ((run=0; run<1; run+=1)); do
  #for prio in "${prios[@]}"; do
  for ((i=0; i<3; i+=1)); do
    /home/nono/.conda/envs/torch_rl/bin/python /home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_gym.py --env-id smac --run-name ${names[i]}-$i-last_buff ${configs[i]} --fixed-buffer --load-buffer-from /home/nono/Documents/Dassault/Water-Bomber-Env/results/full_medium_smac_2023-10-03_12:23:02 --buffer-size 2000000 --train-frequency 1 --total-timesteps 100000 --learning-starts 0 --learning-rate 0.0001 --device cpu --seed $i --last-buffer
  done
done
