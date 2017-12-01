import numpy as np
import pandas as pd

from sklearn.datasets import load_boston

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from clr_regressors import KPlaneRegressor, CLRpRegressor, CLRcRegressor
from clr_ensembles import KPlaneRegressorEnsemble, CLRpRegressorEnsemble, CLRcRegressorEnsemble
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
  boston = load_boston()
  np.random.seed(0)

  X, y = preprocess_data(boston.data, boston.target)

  params = {
    'lr': [Ridge(alpha=1e-5), X, y],
    'ridge 10.0': [Ridge(alpha=10.0), X, y],
  }

  for C in [0.1, 1.0, 10.0, 100.0, 128.0]:
    for g in ['auto', 0.25]:
      for eps in [2 ** (-8), 0.001, 0.1]:
        params['svr C={}, g={}, eps={}'.format(C, g, eps)] = [SVR(C=C, gamma=g, epsilon=eps), X, y]

  for max_depth in [None, 10, 50]:
    for max_features in ['auto', 5]:
      for min_samples_split in [2, 10, 30]:
        for min_samples_leaf in [1, 10, 30]:
          params[
            'rf md={}, mf={}, mss={}, msl={}'.format(
              max_depth,max_features, min_samples_split, min_samples_leaf)
          ] = [RandomForestRegressor(
            n_estimators=100, max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf), X, y]

  # for lr in [None, Ridge(0.1), Ridge(1.0), Ridge(10.0), Lasso(0.1), Lasso(1.0), Lasso(10.0)]
  for f in [True, False]:
    for w in [True, False]:
      for k in [2, 4, 6, 8]:
        for l in [0, 1, 10, 100]:
          params['kplane k={} l={} w={} f={}'.format(k, l, w, f)] = [
            KPlaneRegressor(k, l, weighted=w, fuzzy=f), X, y]
          params['CLS_p k={} l={} w={} f={}'.format(k, l, w, f)] = [
            CLRpRegressor(k, l, weighted=w, fuzzy=f), X, y]
          params['kplane k={} l={} w={} f={} ens=10'.format(k, l, w, f)] = [
            KPlaneRegressorEnsemble(k, l, weighted=w, fuzzy=f), X, y]
          params['CLS_p k={} l={} w={} f={} ens=10'.format(k, l, w, f)] = [
            CLRpRegressorEnsemble(k, l, weighted=w, fuzzy=f), X, y]

  for k in [2, 4, 6, 8]:
    for l in [0, 1, 10, 100]:
      params['CLS_c k={} l={}'.format(k, l)] = [CLRcRegressor(k, l, constr_id=8), X, y]
      params['CLS_c k={} l={} ens=10'.format(k, l)] = [CLRcRegressorEnsemble(k, l, constr_id=8), X, y]

  evaluate_all(params)
