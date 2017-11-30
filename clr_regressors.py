import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from clr import best_clr


class CLRcRegressor(BaseEstimator):
  def __init__(self, num_planes, kmeans_coef,
               constr_id, num_tries=1, clr_lr=None):
    self.num_planes = num_planes
    self.kmeans_coef = kmeans_coef
    self.num_tries = num_tries
    self.constr_id = constr_id
    self.clr_lr = clr_lr

  def fit(self, X, y, init_labels=None, max_iter=100,
          seed=None, verbose=False):
    if seed is not None:
      np.random.seed(seed)
    X, y = check_X_y(X, y)

    constr = np.empty(X.shape[0], dtype=np.int)
    for i, c_id in enumerate(np.unique(X[:, self.constr_id])):
      constr[X[:, self.constr_id] == c_id] = i

    self.labels_, self.models_, _ = best_clr(
      X, y, k=self.num_planes, kmeans_X=self.kmeans_coef,
      constr=constr, max_iter=max_iter, num_tries=self.num_tries,
      lr=self.clr_lr
    )
    # TODO: optimize this
    self.constr_to_label = {}
    for i in range(X.shape[0]):
      self.constr_to_label[X[i, self.constr_id]] = self.labels_[i]

  def predict(self, X):
    check_is_fitted(self, ['labels_', 'models_'])
    X = check_array(X)

    # TODO: optimize this
    test_labels = np.zeros(X.shape[0], np.int)
    for i in range(X.shape[0]):
      test_labels[i] = self.constr_to_label[X[i, self.constr_id]]

    preds = np.empty(X.shape[0])
    for cl_idx in range(self.num_planes):
      if np.sum(test_labels == cl_idx) == 0:
        continue
      y_pred = self.models_[cl_idx].predict(X[test_labels == cl_idx])
      preds[test_labels == cl_idx] = y_pred
    return preds


class KPlaneLabelPredictor(BaseEstimator):
  def __init__(self, num_planes):
    self.num_planes = num_planes
    self.n_classes_ = num_planes

  def fit(self, X, y):
    self.centers_ = np.empty((self.num_planes, X.shape[1]))
    for cl in range(self.num_planes):
      if np.sum(y == cl) == 0:
        # filling with inf empty clusters
        self.centers_[cl] = np.ones(X.shape[1]) * 1e5
        continue
      self.centers_[cl] = np.mean(X[y == cl], axis=0)

  def predict(self, X):
    dst = cdist(self.centers_, X)
    return np.argmin(dst, axis=0)

  def predict_proba(self, X):
    dst = cdist(self.centers_, X)
    return dst.T / np.sum(dst.T, axis=1, keepdims=True)

  def score(self, X, y):
    return np.mean(self.predict(X) == y)


class CLRpRegressor(BaseEstimator):
  def __init__(self, num_planes, kmeans_coef, clr_lr=None,
               num_tries=1, clf=None, weighted=False):
    self.num_planes = num_planes
    self.kmeans_coef = kmeans_coef
    self.num_tries = num_tries
    self.weighted = weighted
    self.clr_lr = clr_lr

    if clf is None:
      # TODO: this is slooooow
      self.clf = RandomForestClassifier(
        n_estimators=20
      )
    else:
      self.clf = clf

  def fit(self, X, y, init_labels=None, max_iter=100,
          seed=None, verbose=False):
    if seed is not None:
      np.random.seed(seed)
    X, y = check_X_y(X, y)
    self.labels_, self.models_, _ = best_clr(
      X, y, k=self.num_planes, kmeans_X=self.kmeans_coef,
      max_iter=max_iter, num_tries=self.num_tries,
      lr=self.clr_lr,
    )
    self.X_ = X
    if verbose:
      label_score = self.get_label_score_()
      print("Label prediction: {:.6f} +- {:.6f}".format(
        label_score.mean(), label_score.std()))
    self.clf.fit(X, self.labels_)

  def get_label_score_(self):
    return cross_val_score(self.clf, self.X_, self.labels_, cv=3).mean()

  def predict(self, X):
    check_is_fitted(self, ['labels_', 'models_'])
    X = check_array(X)

    if self.weighted:
      if self.clf.n_classes_ == self.num_planes:
        planes_probs = self.clf.predict_proba(X)
      else:
        planes_probs = np.zeros((X.shape[0], self.num_planes))
        planes_probs[:, self.clf.classes_] = self.clf.predict_proba(X)
      preds = np.empty((X.shape[0], self.num_planes))
      for cl_idx in range(self.num_planes):
        preds[:,cl_idx] = self.models_[cl_idx].predict(X)
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


class KPlaneRegressor(CLRpRegressor):
  def __init__(self, num_planes, kmeans_coef,
               num_tries=1, weighted=False, clr_lr=None):
    super(KPlaneRegressor, self).__init__(
      num_planes, kmeans_coef,
      num_tries=num_tries,
      clr=KPlaneLabelPredictor(num_planes),
      weighted=weighted, clr_lr=clr_lr,
    )

