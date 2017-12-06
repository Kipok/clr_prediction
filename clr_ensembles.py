from __future__ import print_function
from builtins import range

import numpy as np
from clr_regressors import CLRpRegressor, CLRcRegressor, KPlaneRegressor
from sklearn.base import BaseEstimator


class BaseEnsemble(BaseEstimator):
  def __init__(self, rgr, num_ensembles, **kwargs):
    self.__dict__.update(kwargs)
    self.rgr = rgr
    self.num_ensembles = num_ensembles
    self.rgrs = []
    for i in range(self.num_ensembles):
      self.rgrs.append(self.rgr(**kwargs))

  def fit(self, X, y, init_labels=None, max_iter=100,
          seed=None, verbose=False):
    if seed is not None:
      np.random.seed(seed)
    for i in range(len(self.rgrs)):
      self.rgrs[i].fit(X, y, init_labels, max_iter, verbose=verbose)

  def predict(self, X):
    ans = np.zeros(X.shape[0])
    for i in range(len(self.rgrs)):
      ans += self.rgrs[i].predict(X)
    return ans / len(self.rgrs)


class CLRpRegressorEnsemble(BaseEnsemble):
  def __init__(self, num_planes, kmeans_coef, clr_lr=None, fuzzy=False,
               num_ensembles=10, clf=None, weighted=False):
    super(CLRpRegressorEnsemble, self).__init__(
      CLRpRegressor, num_ensembles, fuzzy=False,
      num_planes=num_planes, kmeans_coef=kmeans_coef,
      clf=clf, weighted=weighted, clr_lr=clr_lr,
    )


class KPlaneRegressorEnsemble(BaseEnsemble):
  def __init__(self, num_planes, kmeans_coef, clr_lr=None,
               num_ensembles=10, weighted=False, fuzzy=False):
    super(KPlaneRegressorEnsemble, self).__init__(
      KPlaneRegressor, num_ensembles, fuzzy=False,
      num_planes=num_planes, kmeans_coef=kmeans_coef,
      weighted=weighted, clr_lr=clr_lr,
    )


class CLRcRegressorEnsemble(BaseEnsemble):
  def __init__(self, num_planes, kmeans_coef,
               constr_id, clr_lr=None, num_ensembles=10):
    super(CLRcRegressorEnsemble, self).__init__(
      CLRcRegressor, num_ensembles,
      num_planes=num_planes, kmeans_coef=kmeans_coef,
      constr_id=constr_id, clr_lr=clr_lr,
    )

