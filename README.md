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
  python src/run_experiments.py --loss-correction-for-others cur-past --clip-correction-after

