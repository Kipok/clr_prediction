nice -n 19 python grid_search.py --dataset=auto-mpg --n_jobs=1
nice -n 19 python grid_search.py --dataset=boston --n_jobs=1
nice -n 19 python grid_search.py --dataset=abalone --n_jobs=1
nice -n 19 python grid_search_ens.py --dataset=boston --n_jobs=-1
nice -n 19 python grid_search_claims.py --eval_rf --run_clrs --eval_algos --eval_best_ens --n_jobs=-1
