import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error as mse_score
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


class KPlaneRegressor(BaseEstimator):
  def __init__(self, num_planes, kmeans_coef, num_tries, weighted=False):
    self.num_planes = num_planes
    self.kmeans_coef = kmeans_coef
    self.num_tries = num_tries
    self.weighted = weighted

  def fit(self, X, y, init_labels=None, max_iter=100, seed=None):
    if seed is not None:
      np.random.seed(seed)
    X, y = check_X_y(X, y)
    self.labels_, self.models_, _ = best_clr(
      X, y, k=self.num_planes, kmeans_X=self.kmeans_coef,
      max_iter=max_iter, num_tries=self.num_tries,
    )
    self.centers_ = np.empty((self.num_planes, X.shape[1]))
    for cl in range(self.num_planes):
      self.centers_[cl] = np.mean(X[self.labels_ == cl], axis=0)

  def predict(self, X):
    check_is_fitted(self, ['labels_', 'models_', 'centers_'])
    X = check_array(X)

    dst = cdist(self.centers_, X)

    if self.weighted:
      planes_probs = dst.T / np.sum(dst.T, axis=1, keepdims=True)
      preds = np.empty((X.shape[0], self.num_planes))
      for cl_idx in range(self.num_planes):
        preds[:,cl_idx] = self.models_[cl_idx].predict(X)
      preds = np.sum(preds * planes_probs, axis=1)
    else:
      test_labels = np.argmin(dst, axis=0)
      preds = np.empty(X.shape[0])
      for cl_idx in range(self.num_planes):
        y_pred = self.models_[cl_idx].predict(X[test_labels == cl_idx])
        preds[test_labels == cl_idx] = y_pred

    return preds

  # TODO: try to average models from different runs?


class CLRpRegressor(BaseEstimator):
  def __init__(self, num_planes, kmeans_coef,
               num_tries=1, clf=None, weighted=False):
    self.num_planes = num_planes
    self.kmeans_coef = kmeans_coef
    self.num_tries = num_tries
    self.weighted = weighted
    if clf is None:
      self.clf = RandomForestClassifier(n_estimators=50)
    else:
      self.clf = clf

  def fit(self, X, y, init_labels=None, max_iter=100, seed=None, verbose=False):
    if seed is not None:
      np.random.seed(seed)
    X, y = check_X_y(X, y)
    self.labels_, self.models_, _ = best_clr(
      X, y, k=self.num_planes, kmeans_X=self.kmeans_coef,
      max_iter=max_iter, num_tries=self.num_tries,
    )
    self.label_score_ = cross_val_score(self.clf, X, self.labels_, cv=3)
    if verbose:
      print("{:.6f} +- {:.6f}".format(
        self.label_score_.mean(), self.label_score_.std()))
    self.clf.fit(X, self.labels_)

  def predict(self, X):
    check_is_fitted(self, ['labels_', 'models_'])
    X = check_array(X)

    if self.weighted:
      planes_probs = self.clf.predict_proba(X)
      preds = np.empty((X.shape[0], self.num_planes))
      for cl_idx in range(self.num_planes):
        preds[:,cl_idx] = self.models_[cl_idx].predict(X)
      preds = np.sum(preds * planes_probs, axis=1)
    else:
      test_labels = self.clf.predict(X)
      preds = np.empty(X.shape[0])
      for cl_idx in range(self.num_planes):
        y_pred = self.models_[cl_idx].predict(X[test_labels == cl_idx])
        preds[test_labels == cl_idx] = y_pred

    return preds


def best_clr(X, y, k, kmeans_X=0.0, max_iter=100, num_tries=10):
  best_obj = 1e9
  for i in range(num_tries):
    labels, models, obj = clr(X, y, k, kmeans_X, max_iter)
    if obj < best_obj:
      best_obj = obj
      best_labels = labels
      best_models = models
  return best_labels, best_models, best_obj


def clr(X, y, k, kmeans_X=0.0, max_iter=100, labels=None, verbose=0):
  if labels is None:
    labels = np.random.choice(k, size=X.shape[0])
  models = [Ridge(alpha=1e-5) for i in range(k)]
  scores = np.empty((X.shape[0], k))
  preds = np.empty((X.shape[0], k))

  for it in range(max_iter):
    # rebuild models
    for cl_idx in range(k):
      models[cl_idx].fit(X[labels == cl_idx], y[labels == cl_idx])
    # reassign points
    for cl_idx in range(k):
      preds[:, cl_idx] = models[cl_idx].predict(X)
      scores[:, cl_idx] = (y - preds[:, cl_idx]) ** 2
      if kmeans_X > 0:
        center = np.mean(X[labels == cl_idx], axis=0)
        scores[:, cl_idx] += kmeans_X * np.sum((X - center) ** 2, axis=1)
    labels_prev = labels.copy()
    labels = np.argmin(scores, axis=1)
    if verbose > 1:
      corr_preds = preds[np.arange(preds.shape[0]), labels]
      print("Iter #{}: obj = {:.6f}, MSE = {:.6f}, r2 = {:.6f}".format(
            it, np.mean(scores[np.arange(preds.shape[0]), labels]),
            mse_score(y, corr_preds), r2_score(y, corr_preds),
      ))
    if np.allclose(labels, labels_prev):
      break
  obj = np.mean(scores[np.arange(preds.shape[0]), labels])
  if verbose == 1:
    corr_preds = preds[np.arange(preds.shape[0]), labels]
    print("Iter #{}: obj = {:.6f}, MSE = {:.6f}, r2 = {:.6f}".format(
          it, obj, mse_score(y, corr_preds), r2_score(y, corr_preds),
    ))
  return labels, models, obj

