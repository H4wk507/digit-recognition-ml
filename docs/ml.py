from sklearn.naive_bayes import GaussianNB
from utils import read_digits
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from collections import defaultdict
import matplotlib.pyplot as plt


def test_neighbors(X, y, x_axis, p: int, filename: str, n_tests: int = 100):
    m = defaultdict(float)
    for _ in range(n_tests):
        for k in x_axis:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
            model = KNeighborsClassifier(n_neighbors=k, p=p)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            m[k] += 100 * accuracy / n_tests
    plt.scatter(x_axis, m.values(), color="orange")
    plt.xlabel("k neighbors")
    plt.ylabel("Accuracy [%]")
    plt.grid()
    plt.xticks(x_axis)
    plt.savefig(filename)

def test_gaussian(X, y, n_tests: int):
    m = 0
    for _ in range(n_tests):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        m += accuracy / n_tests
    print(f"accuracy: {round(m, 3)}")

X, y = read_digits("imgs")
X = X / 255
x_axis = list(range(1, 16, 2))
test_neighbors(X, y, x_axis, 3, "knn3.png")
# test_neighbors(X, y, 1, 101, 2, 100)
# test_gaussian(X, y, 100)
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
# model = KNeighborsClassifier(n_neighbors=3)
#model.fit(X_train, y_train)
#y_pred = model.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)
#print(round(accuracy, 3))
#