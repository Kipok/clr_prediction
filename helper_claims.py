import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error as mse_score
from sklearn.linear_model import LinearRegression
from clr import clr


def gen_clrs(k, kmeans_X, X, y, max_iter, cv_folds=3, n_estimators=10, constr=None):
  kf = KFold(n_splits=cv_folds, random_state=0)
  lr = LinearRegression()

  for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    for j in range(n_estimators):
      clr_out = clr(
        X[train_idx], y[train_idx], k, kmeans_X,
        max_iter=max_iter, lr=lr,
        constr=constr[train_idx] if constr is not None else constr,
      )
      if constr is None:
        file_name = 'saved_runs/k={}_kmeansX={}_fold={}_run={}.pkl'.format(k, kmeans_X, i, j)
      else:
        file_name = 'saved_runs_constr/k={}_kmeansX={}_fold={}_run={}.pkl'.format(k, kmeans_X, i, j)
      with open(file_name, 'wb') as fout:
        pickle.dump(clr_out, fout)


def rmse_score(y1, y2):
  return np.sqrt(mse_score(y1, y2))

# TODO: a lot of code duplication!!

def eval_algo(algo, algo_name, X, y, k, kmeans_X,
              cv_folds=3, n_estimators=10, use_est=False):
  columns = [
    'fit_time_mean', 'fit_time_std', 'fit_time_std_aggr',
    'test_mse_mean', 'test_mse_std', 'test_mse_std_aggr',
    'train_mse_mean', 'train_mse_std', 'train_mse_std_aggr',
    'train_label_acc_mean', 'train_label_acc_std', 'train_label_acc_std_aggr',
    'test_r2_mean', 'test_r2_std', 'test_r2_std_aggr',
    'train_r2_mean', 'train_r2_std', 'train_r2_std_aggr',
    'test_rmse_mean', 'test_rmse_std', 'test_rmse_std_aggr',
    'train_rmse_mean', 'train_rmse_std', 'train_rmse_std_aggr',
    'score_time_mean', 'score_time_std', 'score_time_std_aggr',
  ]
  kf = KFold(n_splits=cv_folds, random_state=0)
  results = pd.DataFrame(columns=columns)

  scores_def = {
    'test_r2': r2_score,
    'test_mse': mse_score,
    'test_rmse': rmse_score
  }

  preds = None

  for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    if preds is None:
      preds = np.empty((test_idx.shape[0], kf.n_splits, n_estimators))
      preds_w = np.empty((test_idx.shape[0], kf.n_splits, n_estimators))

    for j in range(n_estimators):
      file_name = 'saved_runs/k={}_kmeansX={}_fold={}_run={}.pkl'\
              .format(k, kmeans_X, i, j)
      with open(file_name, 'rb') as fin:
        clr_out = pickle.load(fin)
      algo.init_fit(X[train_idx], clr_out[0], clr_out[1])
      preds[:, i, j] = algo.predict(X[test_idx])
      algo.weighted = True
      preds_w[:, i, j] = algo.predict(X[test_idx])
      if use_est is False:
        break

  # usual
  scores_cv = {}
  for sc_name, sc_func in scores_def.items():
    tmp = []
    for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
      tmp.append(sc_func(y[test_idx], preds[:, i, 0]))
    scores_cv[sc_name + '_mean'] = np.mean(tmp)
    scores_cv[sc_name + '_std'] = np.std(tmp)
  results = results.append(pd.Series(scores_cv, name=(algo_name + ' w=False')))

  # weighted
  scores_cv = {}
  for sc_name, sc_func in scores_def.items():
    tmp = []
    for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
      tmp.append(sc_func(y[test_idx], preds_w[:, i, 0]))
    scores_cv[sc_name + '_mean'] = np.mean(tmp)
    scores_cv[sc_name + '_std'] = np.std(tmp)
  results = results.append(pd.Series(scores_cv, name=(algo_name + ' w=True')))

  if use_est:
    # usual ens
    scores_cv = {}
    for sc_name, sc_func in scores_def.items():
      tmp = []
      for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        tmp.append(sc_func(y[test_idx], np.mean(preds[:, i], axis=-1)))
      scores_cv[sc_name + '_mean'] = np.mean(tmp)
      scores_cv[sc_name + '_std'] = np.std(tmp)
    results = results.append(pd.Series(
        scores_cv, name=(algo_name + ' w=False ens={}'.format(n_estimators)))
    )

    # weighted ens
    scores_cv = {}
    for sc_name, sc_func in scores_def.items():
      tmp = []
      for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        tmp.append(sc_func(y[test_idx], np.mean(preds_w[:, i], axis=-1)))
      scores_cv[sc_name + '_mean'] = np.mean(tmp)
      scores_cv[sc_name + '_std'] = np.std(tmp)
    results = results.append(
        pd.Series(scores_cv, name=(algo_name + ' w=True ens={}'.format(n_estimators)))
    )
  return results


def eval_algo_constr(algo, algo_name, X, y, constr, k, kmeans_X,
                     cv_folds=3, n_estimators=10, use_est=False):
  columns = [
    'fit_time_mean', 'fit_time_std', 'fit_time_std_aggr',
    'test_mse_mean', 'test_mse_std', 'test_mse_std_aggr',
    'train_mse_mean', 'train_mse_std', 'train_mse_std_aggr',
    'train_label_acc_mean', 'train_label_acc_std', 'train_label_acc_std_aggr',
    'test_r2_mean', 'test_r2_std', 'test_r2_std_aggr',
    'train_r2_mean', 'train_r2_std', 'train_r2_std_aggr',
    'test_rmse_mean', 'test_rmse_std', 'test_rmse_std_aggr',
    'train_rmse_mean', 'train_rmse_std', 'train_rmse_std_aggr',
    'score_time_mean', 'score_time_std', 'score_time_std_aggr',
  ]
  kf = KFold(n_splits=cv_folds, random_state=0)
  results = pd.DataFrame(columns=columns)

  scores_def = {
    'test_r2': r2_score,
    'test_mse': mse_score,
    'test_rmse': rmse_score
  }

  preds = None

  for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    if preds is None:
      preds = np.empty((test_idx.shape[0], kf.n_splits, n_estimators))

    for j in range(n_estimators):
      file_name = 'saved_runs_constr/k={}_kmeansX={}_fold={}_run={}.pkl'\
              .format(k, kmeans_X, i, j)
      with open(file_name, 'rb') as fin:
        clr_out = pickle.load(fin)
        
      constr_to_label = {}
      for t in range(train_idx.shape[0]):
        constr_to_label[constr[train_idx[t]]] = clr_out[0][t]

      algo.init_fit(clr_out[0], clr_out[1], constr_to_label)
      preds[:, i, j] = algo.predict(X[test_idx], test_constr=constr[test_idx])
      if use_est is False:
        break

  # usual
  scores_cv = {}
  for sc_name, sc_func in scores_def.items():
    tmp = []
    for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
      tmp.append(sc_func(y[test_idx], preds[:, i, 0]))
    scores_cv[sc_name + '_mean'] = np.mean(tmp)
    scores_cv[sc_name + '_std'] = np.std(tmp)
  results = results.append(pd.Series(scores_cv, name=(algo_name)))
  
  if use_est:
    # usual ens
    scores_cv = {}
    for sc_name, sc_func in scores_def.items():
      tmp = []
      for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        tmp.append(sc_func(y[test_idx], np.mean(preds[:, i], axis=-1)))
      scores_cv[sc_name + '_mean'] = np.mean(tmp)
      scores_cv[sc_name + '_std'] = np.std(tmp)
    results = results.append(pd.Series(scores_cv, name=(algo_name + ' ens={}'.format(n_estimators))))

  return results
