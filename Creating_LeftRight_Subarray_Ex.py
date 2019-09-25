import numpy as np

X = np.array([[0, 1, 1], [1, 0, 1], [1, 0, 0]])
feature_split = 1
Xleft = None

for i in range(len(X)):
    if X[i][feature_split] == 0:
        if Xleft is None:
            Xleft = np.vstack([X[i]])
        else:
            Xleft = np.vstack([Xleft, X[i]])

print(Xleft)