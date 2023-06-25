import numpy as np


class NaiveBayesClassifier:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        self.num_features = X.shape[1]
        self.class_probs = np.zeros(self.num_classes)
        self.feature_probs = np.zeros((self.num_classes, self.num_features))
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_probs[i] = len(X_c) / len(X)
            self.feature_probs[i] = np.mean(X_c, axis=0)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0], dtype=np.str_)
        for i, x in enumerate(X):
            posterior_probs = []
            for j in range(self.num_classes):
                prior = np.log(self.class_probs[j])
                likelihood = np.sum(
                    np.log(self.compute_feature_prob(self.feature_probs[j], x))
                )
                posterior = prior + likelihood
                posterior_probs.append(posterior)
            y_pred[i] = self.classes[np.argmax(posterior_probs)]
        return y_pred

    def compute_feature_prob(self, feature_prob, x):
        epsilon = 1e-9
        return feature_prob * x + (1 - feature_prob) * (1 - x + epsilon)

    def accuracy_score(self, y_test, y_pred):
        return sum([y_test[i] == y_pred[i] for i in range(len(y_test))]) / len(
            y_test
        )
