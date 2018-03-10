from Neighbors.library import *
from sklearn.datasets import load_boston

boston = load_boston()

print("Форма массива data для набор boston: {}".format(boston.data.shape))

X, y = mglearn.datasets.load_extended_boston()

print("Форма массива X: {}".format(X.shape))
