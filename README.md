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
python src/run_experiments.py --single-agent True False --nb-runs 3  --env-id water-bomber 
python src/run_experiments.py --add-epsilon True False --add-others-explo True False 
python src/run_experiments.py --rb uniform likely --prio none --filter td_error td-past td-cur-past td-cur cur-past cur