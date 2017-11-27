import numpy as np
from sklearn.metrics import r2_score, mean_squared_error as mse_score


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
      if np.sum(labels == cl_idx) == 0:
        continue
      models[cl_idx].fit(X[labels == cl_idx], y[labels == cl_idx])
    # reassign points
    for cl_idx in range(k):
      preds[:, cl_idx] = models[cl_idx].predict(X)
      scores[:, cl_idx] = (y - preds[:, cl_idx]) ** 2
      if np.sum(labels == cl_idx) == 0:
        continue
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

