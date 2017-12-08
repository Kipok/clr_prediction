from __future__ import print_function
from builtins import range

import pandas as pd


def visualize_results(name, best_only=False):
  res = pd.read_csv(name, index_col=0)
  sort_columns = [
    'fit_time_mean', 'fit_time_std', 'fit_time_std_aggr',
    'test_mse_mean', 'test_mse_std', 'test_mse_std_aggr',
    'train_mse_mean', 'train_mse_std', 'train_mse_std_aggr',
    'train_label_acc_mean', 'train_label_acc_std', 'train_label_acc_std_aggr',
    'test_r2_mean', 'test_r2_std', 'test_r2_std_aggr',
    'train_r2_mean', 'train_r2_std', 'train_r2_std_aggr',
    'test_rmse_mean', 'test_rmse_std', 'test_rmse_std_aggr',
    'train_rmse_mean', 'train_rmse_std', 'train_rmse_std_aggr',
    'score_time_mean', 'score_time_std', 'score_time_std_aggr',
  ]
  results = res[sort_columns]
  results = results.sort_values('test_mse_mean')
  if best_only:
    algos1 = ['CLR_p', 'CLR_c', 'kplane']
    algos2 = ['rf', 'lr', 'svr']
    selected_index = []
    for algo_name in algos2:
      for idx in results.index:
        if algo_name in idx:
          selected_index.append(idx)
          break
    for algo_name in algos1:
      for idx in results.index:
        if algo_name in idx and 'ens' not in idx:
          selected_index.append(idx)
          break
      for idx in results.index:
        if algo_name in idx and 'ens' in idx:
          selected_index.append(idx)
          break
    return results.loc[selected_index].sort_values('test_mse_mean')
  return results

