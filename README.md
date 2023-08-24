# Water-Bomber-Env


 --prio td_error td-past td-cur-past td-cur cur-past cur --env-id water-bomber
 --rb uniform prioritized laber --env-id water-bomber
 --rb uniform prioritized laber --add-epsilon --env-id water-bomber
 --rb uniform prioritized laber --add-epsilon --add-others-explo --env-id water-bomber
 --rb uniform prioritized laber --loss-corrected-for-others --env-id water-bomber
 --n-agents 1 2 3 4 5 --env-id water-bomber 
 --add-epsilon True False --add-others-explo True False
 --single-agent True False  --env-id water-bomber 
 --loss-corrected-for-others True False --env-id water-bomber 