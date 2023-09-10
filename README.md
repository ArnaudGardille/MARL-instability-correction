# Water-Bomber-Env

tensorboard --logdir=/home/nono/Documents/Dassault/Water-Bomber-Env/results/runs/test --port 6007


 --prio td_error td-past td-cur-past td-cur cur-past cur --env-id water-bomber
 --rb uniform prioritized laber --env-id water-bomber
 --rb uniform prioritized laber --add-epsilon --env-id water-bomber
 --rb uniform prioritized laber --add-epsilon --add-others-explo --env-id water-bomber
 --rb uniform prioritized laber --loss-corrected-for-others --env-id water-bomber
 --n-agents 1 2 3 4 5 --env-id water-bomber 
 --add-epsilon True False --add-others-explo True False
 --single-agent True False  --env-id water-bomber 
 --loss-corrected-for-others True False --env-id water-bomber 
  python src/run_experiments.py


  python src/run_experiments.py --loss-correction-for-others none td_error td-past td-cur-past td-cur cur-past cur --rb laber 
  python src/run_experiments.py --loss-correction-for-others none td_error td-past td-cur-past td-cur cur-past cur --rb laber  --sqrt-correction
  python src/run_experiments.py --rb uniform prioritized laber likely
  python src/run_experiments.py --rb  prioritized  likely
  python src/run_experiments.py --rb uniform likely --prio none
  python src/run_experiments.py --loss-correction-for-others cur-past --clip-correction-after 0.1 0.2 0.5 1.0 

python src/run_experiments.py --rb prioritized likely --single-agent True

python src/run_experiments.py -rb uniform likely --prio none --single-agent True 

python src/run_experiments.py --rb uniform likely --prio none --filter td_error td-past td-cur-past td-cur cur-past cur
python src/run_experiments.py --rb laber --prio none td_error td-past td-cur-past td-cur cur-past cur --rb laber  --sqrt-correction

tensorboard --logdir=/home/nono/Documents/Dassault/Water-Bomber-Env/results/ --port 6008

python src/run_experiments.py --single-agent True False



/home/nono/.conda/envs/torch_rl/bin/python /home/nono/Documents/Dassault/Water-Bomber-Env/src/run_experiments.py  --add-epsilon True False --add-others-explo True False --env-id water-bomber 
python src/run_experiments.py --rb uniform likely --prio none --filter td_error td-past td-cur-past td-cur cur-past cur --env-id water-bomber

/home/nono/.conda/envs/torch_rl/bin/python /home/nono/Documents/Dassault/Water-Bomber-Env/src/run_experiments.py --single-agent True False  --env-id water-bomber --n-agents 2 3 4

python src/run_experiments.py --n-agents 6 8 10 12 14  --nb-runs 10 

python src/run_experiments.py --rb uniform prioritized laber likely --nb-runs 1

--nb-runs 100
--nb-runs 3 --env-id water-bomber 

python src/run_experiments.py --loss-correction-for-others none td_error td-past td-cur-past td-cur cur-past cur 
python src/run_experiments.py --rb uniform prioritized laber likely 
python src/run_experiments.py --rb laber --prio none td_error td-past td-cur-past td-cur cur-past cur 
python src/run_experiments.py --rb uniform likely --prio none --filter td_error td-past td-cur-past td-cur cur-past cur

python src/run_experiments.py --add-epsilon True False --add-others-explo True False 
python src/run_experiments.py --rb uniform likely --prio none --filter td_error td-past td-cur-past td-cur cur-past cur

python src/run_experiments.py --rb uniform prioritized laber likely --env-id smac --nb-runs 1
python src/run_experiments.py --loss-correction-for-others none td_error td-past td-cur-past td-cur cur-past cur --nb-runs 100
python src/run_experiments.py --single-agent True False --nb-runs 3  --env-id water-bomber 
 python src/run_experiments.py --add-epsilon True False --add-others-explo True False --nb-runs 100
 python src/run_experiments.py --rb laber --prio none td_error td-past td-cur-past td-cur cur-past cur --nb-runs 100
 python src/run_experiments.py --add-epsilon True False --add-others-explo True False --env-id water-bomber 

  --rb  

  python src/run_experiments.py --rb uniform prioritized laber likely correction --nb-runs 3 --env-id water-bomber 
 python src/run_experiments.py  --rb prioritized likely --filter td-cur-past --nb-runs 10 --env-id water-bomber

python src/run_experiments.py --add-epsilon True False --add-others-explo True False --nb-runs 10 --env-id water-bomber 
python src/run_experiments.py --rb uniform likely --prio none --filter cur-past cur --nb-runs 3 --env-id water-bomber
python src/run_experiments.py  --rb prioritized likely --filter td-cur --nb-runs 10 --env-id water-bomber


python src/run_experiments.py --rb laber --prio none cur cur-past

python src/run_experiments.py --rb uniform laber likely --prio cur --filter cur-past prioritize-big-buffer False 

--loss-correction-for-others cur
python src/run_experiments.py --rb laber --prio none cur cur-past --loss-correction-for-others none cur cur-past  

python src/run_experiments.py --rb laber --prio none cur cur-past --loss-correction-for-others none cur cur-past  --nb-runs 100


python src/run_experiments.py --rb uniform laber likely --prio cur-past --filter cur-past --prioritize-big-buffer False --nb-runs 100

python src/remake_plots.py --name                   

python src/run_experiments.py --rb laber likely --prio cur-past  --filter cur-past --loss-correction-for-others none cur cur-past  

python src/run_experiments.py --rb laber --prio cur-past td_error td-cur-past --prioritize_big_buffer False  

python src/run_experiments.py --rb uniform laber likely --prio cur-past --filter cur-past

--loss_correction_for_others none cur cur-past


python src/run_experiments.py --rb uniform laber likely --prio cur-past --filter cur-past --env-id smac --single-agent True
--prioritize-big-buffer True

/home/nono/.conda/envs/torch_rl/bin/python /home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_gym.py --total-timesteps 1000000
 --buffer-size 20000000


 python src/run_experiments.py --rb laber likely --prio cur-past  --filter  none cur past cur-past --loss-correction-for-others none cur cur-past  
--map 10m_vs_11m 27m_vs_30m 2c_vs_64zg 2s3z 2s_vs_1sc 3s5z 3s5z_vs_3s6z 3s_vs_5z bane_vs_bane corridor mmm mmm2

python src/run_experiments.py --env-id smac --map 10m_vs_11m 27m_vs_30m 2c_vs_64zg 2s3z 2s_vs_1sc 3s5z 3s5z_vs_3s6z 3s_vs_5z bane_vs_bane corridor mmm mmm2 --nb-runs 1 --device cpu


python src/run_experiments.py --rb laber --prio none cur past --loss-correction-for-others none cur cur-past 

python src/run_experiments.py --rb laber --prio none cur past --loss-correction-for-others none cur cur-past --nb-runs 3 --env-id water-bomber
python src/run_experiments.py --rb laber --prio none cur past --loss-correction-for-others none cur cur-past  --nb-runs 100

/home/nono/.conda/envs/torch_rl/bin/python /home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_gym.py --env-id smac --use-state

python src/run_experiments.py --rb uniform laber likely --prio cur-past --filter cur-past --env-id smac --nb-runs 2
python src/run_experiments.py --add-epsilon True False  --env-id smac --nb-runs 3
python src/run_experiments.py --add-others-explo True False --env-id smac --nb-runs 3
python src/run_experiments.py --rb uniform prioritized laber --env-id smac --nb-runs 2

/home/nono/.conda/envs/torch_rl/bin/python /home/nono/Documents/Dassault/Water-Bomber-Env/src/iql_gym.py --single-agent --add-id --env-id smac --map MMM2 --save-model --save-buffer --use-state