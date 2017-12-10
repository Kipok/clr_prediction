from __future__ import print_function
from builtins import range

import numpy as np
import pandas as pd
import sys
import argparse
import os

from scipy.sparse import csc_matrix

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from clr_regressors import KPlaneRegressor, CLRpRegressor, CLRcRegressor
from evaluate import evaluate_all
from helper_claims import gen_clrs, eval_algo, eval_algo_constr
from multiprocessing import Pool


def lambda_gen_clrs(pm):
  return gen_clrs(*pm)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Search parameters')
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--n_jobs', default=1, type=int)
  parser.add_argument('--global_parallel', dest='global_parallel', action='store_true')
  parser.add_argument('--eval_rf', dest='eval_rf', action='store_true')
  parser.add_argument('--run_clrs', dest='run_clrs', action='store_true')
  parser.add_argument('--eval_algos', dest='eval_algos', action='store_true')
  parser.add_argument('--eval_best_ens', dest='eval_best_ens', action='store_true')
  args = parser.parse_args()
  np.random.seed(args.seed)

  data = pd.read_csv('data/patient-claims.csv', index_col=0)
  X = csc_matrix(data.drop(['length', 'provider_id'], axis=1).as_matrix().astype(np.float))
  y = data.length.as_matrix().astype(np.float)

  constr = np.empty(X.shape[0], dtype=np.int)
  for i, c_id in enumerate(np.unique(data.provider_id)):
    constr[data.provider_id == c_id] = i

  results = None

  if args.eval_rf:
    params = {
      'lr': [LinearRegression(), X, y],
      'ridge 10.0': [Ridge(alpha=10.0), X, y],
    }

    n_jobs = 1 if args.global_parallel else args.n_jobs
    for max_depth in [None, 50]:
      for max_features in ['log2', 'sqrt', X.shape[1] // 4]:
        for min_samples_split in [2, 10, 30, 50]:
          for min_samples_leaf in [1, 10, 30, 50]:
            params[
              'rf md={}, mf={}, mss={}, msl={}'.format(
                max_depth,max_features, min_samples_split, min_samples_leaf)
            ] = [RandomForestRegressor(
              n_estimators=30, max_depth=max_depth,
              max_features=max_features, min_samples_leaf=min_samples_leaf,
              min_samples_split=min_samples_split, n_jobs=n_jobs), X, y, 3, 1]

    results = evaluate_all(
      params,
      file_name="results/patient-claims-rf.csv",
      n_jobs=args.n_jobs,
      gl_parallel=args.global_parallel,
    )

  if args.run_clrs:
    if args.n_jobs == 1:
      for k in [2, 4, 6, 8]:
        for l in [0, 1, 10, 100, 1000]:
          gen_clrs(k, l, X, y, max_iter=20, n_estimators=1)
          gen_clrs(k, l, X, y, max_iter=20, n_estimators=1, constr=constr)
    else:
      pms = []
      for k in [2, 4, 6, 8]:
        for l in [0, 1, 10, 100, 1000]:
          for c in [constr, None]:
            pms.append([k, l, X, y, 30, 3, 10, c])
      p = Pool(args.n_jobs)
      p.map(lambda_gen_clrs, pms)
      p.terminate()

  if args.eval_algos:
    for k in [2, 4, 6, 8]:
      for l in [0, 1, 10, 100, 1000]:
        kmeans_X = l
        algo = CLRcRegressor(k, kmeans_X, -1)
        algo_name = 'CLR_c k={} l={}'.format(k, kmeans_X)
        res = eval_algo_constr(algo, algo_name, X, y, constr, k, kmeans_X)
        if results is None:
          results = res
        results = results.append(res)

        algo = KPlaneRegressor(k, kmeans_X, weighted=False)
        algo_name = 'kplane k={} l={}'.format(k, kmeans_X)
        res = eval_algo(algo, algo_name, X, y, k, kmeans_X, n_estimators=3)
        results = results.append(res)

        algo = CLRpRegressor(
          k, kmeans_X, weighted=False,
          clf = RandomForestClassifier(
            n_estimators=50, min_samples_split=50, max_features=50, n_jobs=args.n_jobs,
          ),
        )
        algo_name = 'CLR_p k={} l={}'.format(k, kmeans_X)
        res = eval_algo(algo, algo_name, X, y, k, kmeans_X, n_estimators=3)
        results = results.append(res)
        results.to_csv('results/patient-claims-algos.csv')

  if args.eval_best_ens:
    k, kmeans_X = 8, 1
    algo = CLRcRegressor(k, kmeans_X, -1)
    algo_name = 'CLR_c k={} l={}'.format(k, kmeans_X)
    res = eval_algo_constr(algo, algo_name, X, y, constr, k, kmeans_X, n_estimators=10, use_est=True)
    if results is None:
      results = res
    results = results.append(res)

    k, kmeans_X = 8, 1000
    algo = KPlaneRegressor(k, kmeans_X, weighted=False)
    algo_name = 'kplane k={} l={}'.format(k, kmeans_X)
    res = eval_algo(algo, algo_name, X, y, k, kmeans_X, n_estimators=10, use_est=True)
    results = results.append(res)

    k, kmeans_X = 2, 0
    algo = CLRpRegressor(
      k, kmeans_X, weighted=False,
      clf = RandomForestClassifier(
        n_estimators=50, min_samples_split=50, max_features=50, n_jobs=args.n_jobs,
      ),
    )
    algo_name = 'CLR_p k={} l={}'.format(k, kmeans_X)
    res = eval_algo(algo, algo_name, X, y, k, kmeans_X, n_estimators=10, use_est=True)
    results = results.append(res)
    results.to_csv('results/patient-claims-best-ens.csv')

