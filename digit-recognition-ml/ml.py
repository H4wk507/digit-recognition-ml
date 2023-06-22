from sklearn.naive_bayes import GaussianNB
from utils import read_digits
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from collections import defaultdict

def test_neighbors(X, y, start: int, end: int, step: int, n_tests: int):
    m = defaultdict(float)
    for _ in range(n_tests):
        for k in range(start, end+1, step):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            m[k] += accuracy / n_tests
    for k, v in m.items():
        print(f"{k}: {round(v, 3)}")
    print(f"best k: {max(m, key=m.get)}, accuracy: {round(max(m.values()), 3)}")

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
# test_neighbors(X, y, 1, 101, 2, 100)
test_gaussian(X, y, 100)
#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
#model = KNeighborsClassifier(n_neighbors=3)
#model.fit(X_train, y_train)
#y_pred = model.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)
#print(round(accuracy, 3))
