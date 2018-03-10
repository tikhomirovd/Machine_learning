from Neighbors.library import *
import mglearn

X, y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

print("Linear")
print("Правильность на обучающем наборе: {:.2f}".format(
    lr.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.2f}".format(
    lr.score(X_test, y_test)))

ridge = Ridge().fit(X_train,y_train)
print("Ridge")
print("Правильность на обучающем наборе: {:.2f}".format(
    ridge.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.2f}".format(
    ridge.score(X_test, y_test)))
