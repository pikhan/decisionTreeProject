import numpy as np
import decision_trees as dt

Xbin = np.array([[1, 0, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1]])
Ybin = np.array([[1], [0], [1]])

Xbinval = np.array([[1, 1, 0, 1], [1, 0, 0, 1], [1, 0, 1, 1]])
Ybinval = np.array([[0], [1], [1]])

DT = dt.DT_train_binary(Xbin, Ybin, 2)
test_accB = dt.DT_test_binary(Xbin, Ybin, DT)
print(test_accB) 

DT = dt.DT_train_binary_best(Xbin, Ybin, Xbinval, Ybinval)
test_accB = dt.DT_test_binary(Xbinval, Ybinval, DT)
print(test_accB)

Xreal = np.array([[12, 20, 1.5, 19.2], [10.4, 11.8, 40.7, 91.9], [1.31, 57.2, 80, 21]])
Yreal = np.array([[1], [0], [1]])

Xrealval = np.array([[12, 11, 10, 51], [81, 30, 20, 11], [51, 10, 21, 71]])
Yrealval = np.array([[0], [1], [1]])

DT = dt.DT_train_real(Xreal, Yreal, -1)
test_accR = dt.DT_test_real(Xreal, Yreal, DT)

print(test_accR)

DT = dt.DT_train_real_best(Xreal, Yreal, Xrealval, Yrealval)
test_accB = dt.DT_test_real(Xrealval, Yrealval, DT)
print(test_accB)
