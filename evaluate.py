import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate


def label_prediction_score(model, X, y):
  try:
    return model.label_score_.mean()
  except:
    return np.nan


def evaluate(rgr, X, y, cv_folds=10, cv_times=10,
             aggr_cv=False, n_jobs=1, verbose=False):
  score_dict = {
    'r2': 'r2',
    'mse': 'neg_mean_squared_error',
    'label_acc': label_prediction_score,
  }
  keys = ['test_mse', 'train_mse', 'test_r2', 'train_r2',
          'test_rmse', 'train_rmse', 'fit_time', 'score_time',
          'train_label_acc', 'test_label_acc']
  scores = {key: np.array([]) for key in keys}
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
      if aggr_cv:
        scores[key] = np.append(scores[key], np.mean(val))
      scores[key] = np.append(scores[key], val)
  res_scores = {}
  for key in keys:
    if verbose:
      print('{} = {:.6f} +- {:.6f}'.format(
        key, scores[key].mean(), scores[key].std()))
    res_scores[key + '_mean'] = scores[key].mean()
    res_scores[key + '_std'] = scores[key].std()
  return res_scores


def evaluate_all(run_params):
  results = None
  for name, pm in run_params.items():
    print('Processing: {}'.format(name))
    res_scores = evaluate(*pm)
    if results is None:
      results = pd.DataFrame(columns=res_scores.keys())
    results = results.append(pd.Series(res_scores, name=name))
    results.to_csv('results.csv')
  return results

