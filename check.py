# check 1
# import argparse
#
# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# args = parser.parse_args()
#
# print(args.data_dir)

# check 2
# from utils import data_stream
#
# data_stream = data_stream('./model/poetry/feature.npy', './model/poetry/label.npy', 5388)
# for x, y in data_stream:
#     print(x.shape)
#     break

# check 3
# import numpy as np
# a = [[1, 2, 3, 4],
#      [3, 4, 5, 6]]
# a = np.array(a)
# print(a[:, 0:1])
# print(a[:, 1])
# print(len(a[0]))

# check 4
# dictionary = open('./model/poetry/dictionary.txt', 'r', encoding='utf-8').readlines()
# print(len(dictionary))

# check 5
import pickle
with open('./model/poetry/word2id.pkl', 'rb') as f:
    a = pickle.load(f)
for i in a:
    if i == '\n':
        print("T")
