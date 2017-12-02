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

  params = {}

  for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    params['CLS_best Ridge {}'.format(alpha)] = [
      CLRpRegressorEnsemble(8, 10, weighted=False, fuzzy=False, clr_lr=Ridge(alpha)), X, y
    ]

  for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    params['CLS_best Lasso {}'.format(alpha)] = [
      CLRpRegressorEnsemble(8, 10, weighted=False, fuzzy=False, clr_lr=Lasso(alpha)), X, y
    ]

  evaluate_all(params, name='res_lrs.csv')
