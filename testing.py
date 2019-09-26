import numpy as np
import decision_trees as dt


Xone = np.array([[1, 1, 1, 0, 1, 1, 0], [0,0,1,1,0,1,1], [0,1,0,0,1,0,0]])
Yone = np.array([[1], [1], [0]])

Xtwo = np.array([[1, 1, 0, 1, 1, 0, 1], [1,1,0,0,1,1,0], [0,0,0,1,0,1,1]])
Ytwo = np.array([[1], [1], [0]])

Xtst = np.array([[0, 1, 1, 1, 0, 1, 0], [0,1,1,1,0,0,1], [1,0,0,1,1,0,0]])
Ytst = np.array([[0], [1], [0]])

print("The accuracy of the first is: ", dt.DT_test_binary(Xtst,Ytst,dt.DT_train_binary(Xone,Yone, 5)))
print("The accuracy of the second is: ", dt.DT_test_binary(Xtst, Ytst, dt.DT_train_binary(Xtwo, Ytwo, 5)))
print("The vote of the first is: ", dt.DT_make_prediction(dt.DT_train_binary(Xone,Yone, 5)))
print("The vote of the second is: ", dt.DT_make_prediction(dt.DT_train_binary(Xtwo,Ytwo, 5)))


