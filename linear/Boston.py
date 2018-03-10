from Neighbors.library import *
from linear.Ordinary_list_squares import lr
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
print("\nRidge, alpha=1")
print("Правильность на обучающем наборе: {:.2f}".format(
    ridge.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.2f}".format(
    ridge.score(X_test, y_test)))

ridge10 = Ridge(alpha=10).fit(X_train,y_train)
print("\nRidge, alpha=10")
print("Правильность на обучающем наборе: {:.2f}".format(
    ridge10.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.2f}".format(
    ridge10.score(X_test, y_test)))

ridge01 = Ridge(alpha=0.1).fit(X_train,y_train)
print("\nRidge, alpha=0.1")
print("Правильность на обучающем наборе: {:.2f}".format(
    ridge01.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.2f}".format(
    ridge01.score(X_test, y_test)))

plt.plot(ridge.coef_, 's', label="Гребневая регрессия, alpha=1")
plt.plot(ridge10.coef_, '^', label="Гребневая регрессия, alpha=10")
plt.plot(ridge01.coef_, 'v', label="Гребневая регрессия, alpha=0.1")
plt.plot(lr.coef_, 'o', label="Линейная рагрессия")

plt.xlabel("Индекс коэффициента")
plt.ylabel("Оценка коэффициента")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()

plt.show()