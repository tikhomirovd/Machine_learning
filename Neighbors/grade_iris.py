from library import *

knn = KNeighborsClassifier(n_neighbors=10)
iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],
                                                    iris_dataset['target'],
                                                    random_state=3)

# Создаём data frame из данных в массиве X_train
# Маркируем столбцы, используя строки в iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# Создаём матрицу рассеяния из dataframe, цвет точек задаём из y_train
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
                           hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

knn.fit(X_train, y_train)
X_new = np.array([[10, 10, 10, 10]])
prediction = knn.predict(X_new)
y_pred = knn.predict(X_test)





print(iris_dataset['DESCR'] + '\n...')
print("Названия ответов: \n{}".format(iris_dataset['target_names']))
print("Ключи iris_dataset: \n{}".format(iris_dataset.keys()))
print("Названия признаков: \n{}".format(iris_dataset['feature_names']))
print("Тип массива data: \n{}".format(type(iris_dataset['data'])))
print("Формат массива data: \n{}".format(iris_dataset['data'].shape))
print("Немного из data: \n{}".format(iris_dataset['data'][:5]))
print("Тип массива target: \n{}".format(type(iris_dataset['target'])))
print("Формат массива target: \n{}".format(iris_dataset['target'].shape))
print("Массив target: \n{}".format(iris_dataset['target']))
print("Формат X_train: \n{}".format(X_train.shape))
print("X_train: \n{}".format(X_train))
print(knn)
print("Прогноз: {}".format(prediction))
print("Спрогнозируемая метка {}".format(
    iris_dataset['target_names'][prediction]))
print("Прогнозы для тестового набора:\n{}".format(y_pred))
print("Правильность на тестовом наборе: {:.2f}".format(knn.score(X_test, y_test)))





