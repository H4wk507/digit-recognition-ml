from collections import Counter

import numpy as np


class KNN:
    def __init__(self, k: int = 3, p: int = 2):
        self.k = k
        self.p = p

    @staticmethod
    def minkowski_distance(X1, X2, p) -> float:
        return sum([abs((a - b) ** p) for a, b in zip(X1, X2)]) ** (1 / p)

    def fit(self, X, Y) -> None:
        self.X_train = X
        self.Y_train = Y

    def predict(self, X) -> list[float]:
        return [self.predict_point(x) for x in X]

    def predict_point(self, x) -> float:
        distances = [
            KNN.minkowski_distance(x, x_train, self.p)
            for x_train in self.X_train
        ]
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.Y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def accuracy_score(self, y_test, y_pred):
        return sum([y_test[i] == y_pred[i] for i in range(len(y_test))]) / len(
            y_test
        )
