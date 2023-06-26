import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

DIRNAME = os.path.dirname(os.path.dirname(__file__))
sys.path.append(DIRNAME)


from digit_recognition.bayes import NaiveBayesClassifier
from digit_recognition.knn import KNN
from digit_recognition.utils import read_digits


def test_knn(
    X,
    y,
    k_list: list[int],
    p_list: list[int],
    filename: str,
    n_tests: int = 100,
    plot: bool = False,
    colors: list[str] = [],
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


def test_bayes(X, y, n_tests: int = 100) -> None:
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
