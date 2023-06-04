import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def normalize(arr):
    return arr / 255


def shuffle(arr):
    return arr.sample(frac=1).reset_index(drop=True)


# knn accuracy: 97% with k=3, shuffle and normalize doesn't affect it
df_train = pd.read_csv("mnist_train.csv")
df_test = pd.read_csv("mnist_test.csv")
x_train, y_train = df_train.drop("label", axis=1), df_train["label"]
x_test, y_test = df_test.drop("label", axis=1), df_test["label"]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
