import numpy as np
from clr_regressors import CLRpRegressor, CLRcRegressor, KPlaneRegressor
from sklearn.base import BaseEstimator


class BaseEnsemble(BaseEstimator):
  def __init__(self, rgr, **kwargs):
    self.__dict__.update(kwargs)
    self.rgr = rgr
    self.rgrs = []
    for i in range(self.num_tries):
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
  def __init__(self, num_planes, kmeans_coef,
               num_tries=10, clf=None, weighted=False):
    super(CLRpRegressorEnsemble, self).__init__(
      CLRpRegressor,
      num_planes=num_planes, kmeans_coef=kmeans_coef,
      num_tries=num_tries, clf=clf, weighted=weighted
    )


class KPlaneRegressorEnsemble(BaseEnsemble):
  def __init__(self, num_planes, kmeans_coef,
               num_tries=10, weighted=False):
    super(KPlaneRegressorEnsemble, self).__init__(
      KPlaneRegressor,
      num_planes=num_planes, kmeans_coef=kmeans_coef,
      num_tries=num_tries, weighted=weighted,
    )


class CLRcRegressorEnsemble(BaseEnsemble):
  def __init__(self, num_planes, kmeans_coef,
               constr, num_tries=10):
    super(CLRcRegressorEnsemble, self).__init__(
      CLRpRegressor, constr=constr,
      num_planes=num_planes, kmeans_coef=kmeans_coef,
      num_tries=num_tries, clf=clf, weighted=weighted
    )

