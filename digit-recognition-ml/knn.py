from itertools import Counter


class KNN:
    def __init__(self, k: int = 3):
        self.k = k 

    @staticmethod
    def euclidean_distance(X1, X2) -> float:
        return sum([(a - b) ** 2 for a, b in zip(X1, X2)]) ** 0.5
    
    @staticmethod
    def argsort(dat) -> list[int]:
        return sorted(range(len(dat)), key=dat.__getitem__)

    def fit(self, X, Y) -> None:
        self.X_train = X
        self.Y_train = Y

    def predict(self, X) -> list[float]:
        return [self.predict_point(x) for x in X]

    def predict_point(self, x) -> float:
        distances = [KNN.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = KNN.argsort(distances)[:self.k]
        k_nearest_labels = [self.Y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def accuracy_score(self, y_test, y_pred):
        return sum([y_test[i] == y_pred[i] for i in range(len(y_test))]) / len(y_test)