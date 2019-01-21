# check 1
# import argparse
#
# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# args = parser.parse_args()
#
# print(args.data_dir)

# check 2
from utils import data_stream

data_stream = data_stream('./model/poetry/feature.npy', './model/poetry/label.npy', 5388)
for x, y in data_stream:
    print(x.shape)
    break