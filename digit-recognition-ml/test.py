from collections import defaultdict

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sb
import pandas as pd

from utils import read_digits
from bayes import NaiveBayesClassifier
from knn import KNN


def test_knn(
    X,
    y,
    k_list: list[int],
    p_list: list[int],
    filename: str,
    n_tests: int = 100,
    plot: bool = False,
    colors: list[str] = []
) -> None:
    m2 = {}
    for p in p_list:
        m = defaultdict(float)
        for _ in range(n_tests):
            for k in k_list:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, train_size=0.75
                )
                model = KNeighborsClassifier(n_neighbors=k, p=p)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                m[k] += 100 * accuracy / n_tests
        m2[p] = m
    if plot:
        assert len(colors) == len(p_list)
        for i, (k, v) in enumerate(m2.items()):
            plt.scatter(k_list, v.values(), label=f"p={k}", color=colors[i])
        plt.grid()
        plt.xlabel("k neighbors")
        plt.ylabel("Accuracy [%]")
        plt.legend([f"p = {p}" for p in p_list], loc="upper right")
        plt.xticks(k_list)
        plt.savefig(filename)
    else:
        print(m2)


def test_bayes(X, y, n_tests: int) -> None:
    acc = 0
    for _ in range(n_tests):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.75
        )
        model = NaiveBayesClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = model.accuracy_score(y_test, y_pred)
        acc += 100 * accuracy / n_tests
    print(f"accuracy: {round(acc, 3)}")




def error_rate(confusion_matrix):
    a = confusion_matrix
    b = a.sum(axis=1)
    df = []
    for i in range(0, 10):
        temp = 1 - a[i][i] / b[i]
        df.append(temp)

    df = pd.DataFrame(df)
    df.columns = ["% Error rate"]
    return df * 100


X, y = read_digits("imgs")
X = X / 255
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)
model = KNeighborsClassifier(n_neighbors=1, p=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.subplots(figsize=(10, 6))
sb.heatmap(cm, annot = True, fmt = 'g')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
# print(error_rate(cm))