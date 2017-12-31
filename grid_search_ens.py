from __future__ import print_function
from builtins import range

import numpy as np
import pandas as pd
import sys
import argparse
import os

from sklearn.datasets import load_boston

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from clr_regressors import KPlaneRegressor, CLRpRegressor, CLRcRegressor, RegressorEnsemble
from evaluate import evaluate_all


def preprocess_data(X, y):
  X -= np.min(X, axis=0, keepdims=True)
  X /= np.max(X, axis=0, keepdims=True) / 2.0
  X -= 1.0
  shuffle_idx = np.random.choice(X.shape[0], X.shape[0], replace=False)
  X = X[shuffle_idx]
  y = y[shuffle_idx]
  return X, y


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Search parameters')
  parser.add_argument('--dataset', required=True)
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--n_jobs', default=1, type=int)
  parser.add_argument('--global_parallel', dest='global_parallel', action='store_true')
  args = parser.parse_args()
  np.random.seed(0)

  if args.dataset == 'boston':
    boston = load_boston()
    X = boston.data
    y = boston.target
    constr_id = 8
  elif args.dataset == 'abalone':
    abalone_data = pd.read_csv('data/abalone.data', header=None)
    with open('data/abalone.names', 'r') as fin:
      abalone_descr = fin.read()
    X = pd.get_dummies(abalone_data.iloc[:,:-1], columns=[0]).as_matrix().astype(np.float)
    X = np.hstack((X, (np.digitize(X[:, 2], np.linspace(0.1, 0.2, 10)))[:,np.newaxis]))
    y = abalone_data.iloc[:, 8].as_matrix().astype(np.float)
    constr_id = 10
  elif args.dataset == 'auto-mpg':
    data = pd.read_csv('data/auto-mpg.data', header=None, sep='\s+', na_values='?')
    data = data.dropna()
    X = pd.get_dummies(data.iloc[:,1:-1], columns=[7]).as_matrix().astype(np.float)
    y = data[0].as_matrix().astype(np.float)
    constr_id = 5
  else:
    print("Dataset is not supported")
    sys.exit(0)
  X, y = preprocess_data(X, y)

  params = {}

  for k in [2, 8]:
    for ens in range(1, 21):
      params['kplane k={} ens={}'.format(k, ens)] = [
        RegressorEnsemble(KPlaneRegressor(k, 100), n_estimators=ens), X, y, 10, 20]
      params['CLR_p k={} ens={}'.format(k, ens)] = [
        RegressorEnsemble(CLRpRegressor(k, 10, weighted=True), n_estimators=ens), X, y, 10, 20]
      params['CLR_c k={} ens={}'.format(k, ens)] = [
        RegressorEnsemble(CLRcRegressor(k, 10, constr_id=constr_id), n_estimators=ens), X, y, 10, 20]

  results = evaluate_all(
    params,
    file_name="results_ens/{}.csv".format(args.dataset),
    n_jobs=args.n_jobs,
    gl_parallel=args.global_parallel,
  )
