import pandas as pd
import argparse


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Combine datasets')
  parser.add_argument('names', nargs='+', default=[],
                      help='List of datasets to combine')
  parser.add_argument('--out_name', default='results_complete.csv',
                      help='Name of the final csv')
  args = parser.parse_args()

  res_complete = pd.read_csv(args.names[0], index_col=0)
  for name in args.names[1:]:
    res_cur = pd.read_csv(name, index_col=0)
    res_complete = res_complete.append(res_cur)
  res_complete.to_csv(args.out_name)

