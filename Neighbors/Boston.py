from library import *


boston = load_boston()

print("Форма массива data для набор boston: {}".format(boston.data.shape))

X, y = mglearn.datasets.load_extended_boston()

print("Форма массива X: {}".format(X.shape))
