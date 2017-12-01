import numpy as np
from sklearn.metrics import r2_score, mean_squared_error as mse_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.base import clone


def reassign_labels(scores, constr):
  if constr is None:
    return np.argmin(scores, axis=1)
  labels = np.empty(scores.shape[0], dtype=np.int)
  for c_id in range(constr.max() + 1):
    labels[constr == c_id] = np.argmin(np.mean(scores[constr == c_id], axis=0))
  return labels


def fuzzy_clr(X, y, k, kmeans_X=0.0,
              max_iter=1000, verbose=0, lr=None):
  if lr is None:
    lr = LinearRegression(fit_intercept=False)
    X = np.hstack((X, np.ones((X.shape[0], 1))))
  models = [clone(lr) for i in range(k)]
  q = np.random.rand(X.shape[0], k)
  q /= np.sum(q, axis=1, keepdims=True)

  sigma_sq = np.empty(k)
  lmbda = np.empty(k)
  centers = np.empty((k, X.shape[1]))
  probs = np.empty((X.shape[0], k))

  for it in range(max_iter):
    # M step
    for cl_idx in range(k):
      q_sqrt = np.sqrt(q[:, cl_idx])
      q_sum = np.sum(q[:, cl_idx])
      models[cl_idx].fit(q_sqrt[:,np.newaxis] * X, q_sqrt * y)
      centers[cl_idx] = np.sum(q[:, cl_idx:cl_idx+1] * X, axis=0) / q_sum
      lmbda[cl_idx] = q_sum / X.shape[0]
      sigma_sq[cl_idx] = np.sum(q[:, cl_idx] * (
        (y - models[cl_idx].predict(X)) ** 2 +
        kmeans_X * np.sum((X - centers[cl_idx]) ** 2, axis=1)
      ))
      sigma_sq[cl_idx] /= q_sum
    # E step
    q_prev = q.copy()
    for cl_idx in range(k):
      probs[:, cl_idx] = np.exp((
        -(y - models[cl_idx].predict(X)) ** 2 -
        kmeans_X * np.sum((X - centers[cl_idx]) ** 2, axis=1)
      ) / (2 * sigma_sq[cl_idx])) / np.sqrt(np.pi * 2.0 * sigma_sq[cl_idx])
      q[:, cl_idx] = lmbda[cl_idx] * probs[:, cl_idx]
    q /= q.sum(axis=1, keepdims=True)

#    assert(np.allclose(np.sum(q, axis=1), 1.0))
#    assert(np.all(sigma_sq > 0))
#    assert(np.allclose(np.sum(lmbda), 1.0))

    if verbose > 1:
      loglike = -np.sum(np.log(np.sum(lmbda * probs, axis=1)))
      print("Iter #{}: loglike = {:.6f}".format(it, loglike))

    if np.allclose(q_prev, q, atol=1e-5):
      break
  loglike = -np.sum(np.log(np.sum(lmbda * probs, axis=1)))
  if verbose == 1:
      print("Iter #{}: loglike = {:.6f}".format(it, loglike))
  labels = np.argmax(q, axis=1)
  return labels, models, loglike


def clr(X, y, k, kmeans_X=0.0, constr=None, lr=None,
        max_iter=1000, labels=None, verbose=0):
  if labels is None:
    labels = np.random.choice(k, size=X.shape[0])
  if lr is None:
    lr = Ridge(alpha=1e-5)
  models = [clone(lr) for i in range(k)]
  scores = np.empty((X.shape[0], k))
  preds = np.empty((X.shape[0], k))

  for it in range(max_iter):
    # rebuild models
    for cl_idx in range(k):
      if np.sum(labels == cl_idx) == 0:
        continue
      models[cl_idx].fit(X[labels == cl_idx], y[labels == cl_idx])
    # reassign points
    for cl_idx in range(k):
      preds[:, cl_idx] = models[cl_idx].predict(X)
      scores[:, cl_idx] = (y - preds[:, cl_idx]) ** 2
      if np.sum(labels == cl_idx) == 0:
        if verbose > 0:
          print("Cluster vanished! Assigning random point")
#        labels[np.random.choice(X.shape[0])] = cl_idx
        continue
      if kmeans_X > 0:
        center = np.mean(X[labels == cl_idx], axis=0)
        scores[:, cl_idx] += kmeans_X * np.sum((X - center) ** 2, axis=1)
    labels_prev = labels.copy()
    labels = reassign_labels(scores, constr)
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


def best_clr(X, y, k, fuzzy=False, num_tries=10, **kwargs):
  clr_func = fuzzy_clr if fuzzy else clr
  best_obj = 1e9
  for i in range(num_tries):
    labels, models, obj = clr_func(X, y, k, **kwargs)
    if obj < best_obj:
      best_obj = obj
      best_labels = labels
      best_models = models
  return best_labels, best_models, best_obj

