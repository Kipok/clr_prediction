from __future__ import print_function
from builtins import range
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
import time
from multiprocessing import Pool


def label_prediction_score(model, X, y):
  try:
    return model.get_label_score_()
  except:
    return np.nan


def evaluate(rgr, X, y, cv_folds=10, cv_times=5,
             n_jobs=1, verbose=False):
  score_dict = {
    'r2': 'r2',
    'mse': 'neg_mean_squared_error',
    'label_acc': label_prediction_score,
  }
  keys = ['test_mse', 'train_mse', 'test_r2', 'train_r2',
          'test_rmse', 'train_rmse', 'fit_time', 'score_time',
          'train_label_acc', 'test_label_acc']
  scores = {key: np.array([]) for key in keys}
  scores_aggr = {key: np.array([]) for key in keys}
  for i in range(cv_times):
    shuffle_idx = np.random.choice(X.shape[0], X.shape[0], replace=False)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    cur_scores = cross_validate(rgr, X, y, cv=cv_folds, scoring=score_dict,
                                n_jobs=n_jobs, return_train_score=True)
    for key in scores.keys():
      if key.endswith('rmse'):
        val = np.sqrt(-cur_scores[key[:-4] + 'mse'])
      elif key.endswith('mse'):
        val = -cur_scores[key]
      else:
        val = cur_scores[key]
      scores_aggr[key] = np.append(scores_aggr[key], np.mean(val))
      scores[key] = np.append(scores[key], val)
  res_scores = {}
  for key in keys:
    if verbose:
      print('{} = {:.6f} +- {:.6f}'.format(
        key, scores[key].mean(), scores[key].std()))
    res_scores[key + '_mean'] = scores[key].mean()
    res_scores[key + '_std'] = scores[key].std()
    res_scores[key + '_std_aggr'] = scores_aggr[key].std()
  return res_scores


def evaluate_all(run_params, file_name='results.csv',
                 n_jobs=1, gl_parallel=False):
  if not gl_parallel:
    return evaluate_seq(run_params, file_name, n_jobs)
  else:
    return evaluate_par(run_params, file_name, n_jobs)


def lambda_eval(pm):
  return evaluate(*pm)


def evaluate_par(run_params, file_name='results.csv', n_jobs=4):
  results = None
  p = Pool(n_jobs)
  res_scores = p.map(lambda_eval, run_params.values())
  p.terminate()

  for name, res_score in zip(run_params.keys(), res_scores):
    if results is None:
      results = pd.DataFrame(columns=res_score.keys())
    results = results.append(pd.Series(res_score, name=name))
  results.to_csv(file_name)
  return results


def evaluate_seq(run_params, file_name='results.csv', n_jobs=4):
  results = None
  total = len(run_params)
  for i, (name, pm) in enumerate(run_params.items()):
    print('Processing {}/{}: {}'.format(i+1, total, name), end="\r")
    sys.stdout.flush()
    tm = time.time()
    res_scores = evaluate(*pm, n_jobs=n_jobs)
    print('Processed {}/{}: {} [mse = {:.6f}, time = {:.2f}s]'.format(
      i+1, total, name, res_scores['test_mse_mean'], time.time() - tm))
    sys.stdout.flush()
    if results is None:
      results = pd.DataFrame(columns=res_scores.keys())
    results = results.append(pd.Series(res_scores, name=name))
    results.to_csv(file_name)
  return results

