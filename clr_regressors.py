from __future__ import print_function
from builtins import range

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
# from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import pairwise_distances as cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from clr import best_clr


class CLRcRegressor(BaseEstimator):
  def __init__(self, num_planes, kmeans_coef, constr_id,
               num_tries=1, clr_lr=None, max_iter=5):
    self.num_planes = num_planes
    self.kmeans_coef = kmeans_coef
    self.num_tries = num_tries
    self.constr_id = constr_id
    self.clr_lr = clr_lr
    self.max_iter = max_iter

  def fit(self, X, y, init_labels=None,
          seed=None, verbose=False):
    if seed is not None:
      np.random.seed(seed)

    constr = np.empty(X.shape[0], dtype=np.int)
    for i, c_id in enumerate(np.unique(X[:, self.constr_id])):
      constr[X[:, self.constr_id] == c_id] = i

    self.labels_, self.models_, _, _ = best_clr(
      X, y, k=self.num_planes, kmeans_X=self.kmeans_coef,
      constr=constr, max_iter=self.max_iter, num_tries=self.num_tries,
      lr=self.clr_lr,
    )
    # TODO: optimize this
    self.constr_to_label = {}
    for i in range(X.shape[0]):
      self.constr_to_label[X[i, self.constr_id]] = self.labels_[i]

  def init_fit(self, labels, models, constr_to_label):
    self.labels_ = labels
    self.models_ = models
    self.constr_to_label = constr_to_label

  def predict(self, X, test_constr=None):
    check_is_fitted(self, ['labels_', 'models_'])

    if test_constr is None:
      test_constr = X[:, self.constr_id]

    # TODO: optimize this
    test_labels = np.zeros(X.shape[0], np.int)
    for i in range(X.shape[0]):
      test_labels[i] = self.constr_to_label[test_constr[i]]

    preds = np.empty(X.shape[0])
    for cl_idx in range(self.num_planes):
      if np.sum(test_labels == cl_idx) == 0:
        continue
      y_pred = self.models_[cl_idx].predict(X[test_labels == cl_idx])
      preds[test_labels == cl_idx] = y_pred
    return preds


class FuzzyCLRRegressor(BaseEstimator):
  def __init__(self, num_planes, kmeans_coef,
               clr_lr=None, num_tries=1):
    self.num_planes = num_planes
    self.kmeans_coef = kmeans_coef
    self.num_tries = num_tries
    self.clr_lr = clr_lr

  def fit(self, X, y, init_labels=None, max_iter=20,
          seed=None, verbose=False):
    if seed is not None:
      np.random.seed(seed)
    self.labels_, self.models_, self.weights_, _ = best_clr(
      X, y, k=self.num_planes, kmeans_X=self.kmeans_coef,
      max_iter=max_iter, num_tries=self.num_tries,
      lr=self.clr_lr, fuzzy=True
    )
    self.X_ = X

  def predict(self, X):
    check_is_fitted(self, ['labels_', 'models_', 'weights_'])

    preds = np.empty((X.shape[0], self.num_planes))
    for cl_idx in range(self.num_planes):
      preds[:, cl_idx] = self.models_[cl_idx].predict(X)
    preds = np.sum(preds * self.weights_, axis=1)
    return preds


class CLRpRegressor(BaseEstimator):
  def __init__(self, num_planes, kmeans_coef, clr_lr=None, max_iter=5,
               num_tries=1, clf=None, weighted=False, fuzzy=False):
    self.num_planes = num_planes
    self.kmeans_coef = kmeans_coef
    self.num_tries = num_tries
    self.weighted = weighted
    self.clr_lr = clr_lr
    self.fuzzy = fuzzy
    self.max_iter = max_iter

    if clf is None:
      self.clf = RandomForestClassifier(n_estimators=20)
    else:
      self.clf = clf

  def fit(self, X, y, init_labels=None,
          seed=None, verbose=False):
    if seed is not None:
      np.random.seed(seed)
    self.labels_, self.models_, _, _ = best_clr(
      X, y, k=self.num_planes, kmeans_X=self.kmeans_coef,
      max_iter=self.max_iter, num_tries=self.num_tries,
      lr=self.clr_lr, fuzzy=self.fuzzy
    )
    self.X_ = X
    if verbose:
      label_score = self.get_label_score_()
      print("Label prediction: {:.6f} +- {:.6f}".format(
        label_score.mean(), label_score.std()))
    if np.unique(self.labels_).shape[0] == 1:
      self.labels_[0] = 1 if self.labels_[0] == 0 else 0
    self.clf.fit(X, self.labels_)

  def init_fit(self, X, labels, models):
    self.labels_ = labels
    self.models_ = models
    self.X_ = X
    self.clf.fit(X, self.labels_)

  def get_label_score_(self):
    return cross_val_score(self.clf, self.X_, self.labels_, cv=3).mean()

  def predict(self, X):
    check_is_fitted(self, ['labels_', 'models_'])

    if self.weighted:
      if 'n_classes_' in self.clf.__dict__ and self.clf.n_classes_ == self.num_planes:
        planes_probs = self.clf.predict_proba(X)
      else:
        planes_probs = np.zeros((X.shape[0], self.num_planes))
        planes_probs[:, self.clf.classes_] = self.clf.predict_proba(X)
      preds = np.empty((X.shape[0], self.num_planes))
      for cl_idx in range(self.num_planes):
        preds[:, cl_idx] = self.models_[cl_idx].predict(X)
      preds = np.sum(preds * planes_probs, axis=1)
    else:
      test_labels = self.clf.predict(X)
      preds = np.empty(X.shape[0])
      for cl_idx in range(self.num_planes):
        if np.sum(test_labels == cl_idx) == 0:
          continue
        y_pred = self.models_[cl_idx].predict(X[test_labels == cl_idx])
        preds[test_labels == cl_idx] = y_pred
    return preds


class KPlaneLabelPredictor(BaseEstimator):
  def __init__(self, num_planes, weight_mode='kplane'):
    self.num_planes = num_planes
    self.n_classes_ = num_planes
    self.weight_mode = weight_mode

  def fit(self, X, y):
    if self.weight_mode == 'size':
      self.weights = np.empty(self.num_planes)
      for cl in range(self.num_planes):
        self.weights[cl] = np.sum(y == cl)
      self.weights /= np.sum(self.weights)
    else:
      self.centers_ = np.empty((self.num_planes, X.shape[1]))
      for cl in range(self.num_planes):
        if np.sum(y == cl) == 0:
          # filling with inf empty clusters
          self.centers_[cl] = np.ones(X.shape[1]) * 1e5
          continue
        self.centers_[cl] = np.mean(X[y == cl], axis=0)

  def predict(self, X):
    if self.weight_mode == 'size':
      probs = self.predict_proba
      return np.argmax(probs)
    dst = cdist(self.centers_, X)
    return np.argmin(dst, axis=0)

  def predict_proba(self, X):
    if self.weight_mode == 'size':
      return self.weights
    dst = cdist(self.centers_, X)
    return dst.T / np.sum(dst.T, axis=1, keepdims=True)

  def score(self, X, y):
    return np.mean(self.predict(X) == y)


class KPlaneRegressor(CLRpRegressor):
  def __init__(self, num_planes, kmeans_coef, fuzzy=False, max_iter=5,
               num_tries=1, weighted=False, clr_lr=None):
    weighted_param = True if weighted == 'size' else weighted
    super(KPlaneRegressor, self).__init__(
      num_planes, kmeans_coef,
      num_tries=num_tries, fuzzy=fuzzy, max_iter=max_iter,
      clf=KPlaneLabelPredictor(num_planes, weight_mode=weighted),
      weighted=weighted_param, clr_lr=clr_lr,
    )


class RegressorEnsemble(BaseEstimator):
  def __init__(self, rgr, n_estimators=10):
    self.rgr = rgr
    self.n_estimators = n_estimators
    self.rgrs = []
    for i in range(self.n_estimators):
      self.rgrs.append(clone(self.rgr))

  def fit(self, X, y, init_labels=None,
          seed=None, verbose=False):
    if seed is not None:
      np.random.seed(seed)
    for i in range(self.n_estimators):
      self.rgrs[i].fit(X, y, init_labels, verbose=verbose)

  def predict(self, X):
    ans = np.zeros(X.shape[0])
    for i in range(self.n_estimators):
      ans += self.rgrs[i].predict(X)
    return ans / len(self.rgrs)

