from library import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

plt.rc('font', family='Verdana')

X, y = mglearn.datasets.make_forge()
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

print("Прогнозы на тестовом наборе: {}".format(clf.predict(X_test)))
print("Правильность на тестовом наборе: {:.2f}".format(clf.score(X_test, y_test)))


fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("Количество соседей:{}".format(n_neighbors))
    ax.set_xlabel("Признак 0")
    ax.set_ylabel("Признак 1")
axes[0].legend(loc=3)



#
# plt.legend(["Класс 0", "Класс 1"], loc=4)
# plt.xlabel("Первый признак")
# plt.ylabel("Второй признак")
# mglearn.plots.plot_knn_classification(n_neighbors=3)







#
plt.show()