# Ibraheem Khan and Matthew Alighchi
# CS 491: Project 1 - Decision Trees
# Emily Hand - UNR

#Numpy is imported for log functions

import numpy as np

class BinaryNode(object):
    def __init__(self):
        self.left_child = None #the left child of the node, another node
        self.right_child = None #the right child of the node, another node
        self.samples = None #a 1D integer array of training sample indices "in" the node
        self.feature_list = None #a 1D integer array of features we have not yet split upon on the branch
        self.feature_split = None #an integer, the feature index for which we split on at the node
        self.label_prediction = None #an integer, 1 for yes, 0 for no the label prediction our DT will output at the particular node

class RealNode(object):
    def __init__(self):
        self.left_child = None #the left child of the node, another node
        self.right_child = None #the right child of the node, another node
        self.samples = None #a 1D integer array of training sample indices "in" the node
        self.feature_list = None #a 3D array of triples (feature_index, feature_value, feature_sign) for which we have not yet split upon on the branch
        self.feature_split = None #an integer, the feature index f3or which we split on at the node
        self.feature_split_value = None #a double, the value for which we split the feature on
        self.feature_split_sign = None #a string, less or leq resp. corresponding to either "<" or "<=", for which we split the feature value with
        self.label_prediction = None #an integer, 1 for yes, 0 for no the label prediction our DT will output at the particular node


#This is a modular function that will calculate the sum of a list of arguments.

def sigma(*args):
    if args is None:
        print('\n The sum input is invalid \n')
    else:
        summer = 0
        for arg in args:
            summer = summer + arg
        return summer



#This is a modular function that will calculate the entropy of the whole data set

def entropy(Y = None):
    if Y is None:
        print('\n The label set input is None, please correct. \n')
    count_yes = 0
    count_no = 0
    for i in range(len(Y[0])):
        if Y[0][i] == 0:
            count_no += 1
        if Y[0][i] == 1:
            count_yes += 1
        else:
            print('\n Label Set has characters other than 0 or 1, please standardize notation. \n')
    prob_yes = count_yes/len(Y[0])
    prob_no = count_no/len(Y[0])
    H = -1*sigma(prob_yes*np.log2(prob_yes), prob_no*np.log2(prob_no))
    return H

def information_gain():

#This function outputs the depth of the given node
def depth(root, node):

def DT_train_binary(X = None, Y = None, max_depth = -2):
    if max_depth <= -2:
        print('\n Please enter a correct max_depth integer \n')
    if X is None:
        print('\n The X 2D Array is empty. \n')
    if Y is None:
        print('\n The Y 2D Array is empty. \n')



def DT_test_binary(X = None, Y = None, DT = None):
    if X is None:
        print('\n The X 2D Array is empty. \n')
    if Y is None:
        print('\n The Y 2D Array is empty. \n')
    if DT is not None:
        if (DT.left_child is None) and (DT.right_child is None):
            accuracy = 0
            for labels in Y:
                if Y[labels] == DT.label_prediction:
                    accuracy+=1
            return accuracy / len(Y[0])
        else:
            Xleft = None
            Yleft = None                 #fixed
            Xright = None
            Yright = None
            for sample in range(len(X)):
                if X[sample][DT.feature_split] == 0:
                    if Xleft or Yleft is None:
                        Xleft = np.vstack([X[sample]])
                        Yleft = np.vstack([Y[sample]])
                    else:
                        Xleft = np.vstack([Xleft, X[sample]])
                        Yleft = np.vstack([Yleft, Y[sample]])
                else:
                    if Xright or Yright is None:
                        Xright = np.vstack([X[sample]])
                        Yright = np.vstack([Y[sample]])
                    else:
                        Xright = np.vstack([Xright, X[sample]])
                        Yright = np.vstack([Yright, Y[sample]])

            return DT_test_binary(Xleft, Yleft, DT.left_child) + DT_test_binary(Xright, Yright, DT.right_child)
    else:
        return 0

def maxDepth(DT = None):
    if DT is None:
        return 0 ;  
    else : 
        lDepth = maxDepth(DT.left_child)
        rDepth = maxDepth(DT.right_child)
  
        # Use the larger one 
        if (lDepth > rDepth): 
            return lDepth+1
        else: 
            return rDepth+1

def DT_train_binary_best(X_train = None, Y_train = None, X_val = None, Y_val = None):
    DT_best = DT_train_binary(X_train, Y_train, -1)
    acc_best = DT_test_binary(X_val, Y_val, DT_best)
    depth = maxDepth(DT_best)

    for x in range(depth):
        DT_curr = DT_train_binary(X_train, Y_train, x)
        acc_curr = DT_test_binary(X_val, Y_val, DT_curr)
        if acc_curr >= acc_best:
            acc_best = acc_curr
            DT_best = DT_curr

    return DT_best



def DT_make_prediction(x = None, DT = None): #assumes DT is binary (I believe this is correct)
    if (DT.left_child is None) and (DT.right_child is None):
        return DT.label_prediction
    if x is None:
        print('\n Please enter a correct sample index. \n')
    if x[DT.feature_split] == 0:
        return DT_make_prediction(x, DT.left_child)
    else:
        return DT_make_prediction(x, DT.right_child)

def DT_train_real(X = None, Y = None, max_depth = -2):
    if max_depth <= -2:
        print('\n Please enter a correct max_depth integer \n')

def DT_test_real(X = None, Y = None, DT = None):
    if X is None:
        print('\n The X 2D Array is empty. \n')
    if Y is None:
        print('\n The Y 2D Array is empty. \n')
    if DT is not None:
        if (DT.left_child is None) and (DT.right_child) is None:
            accuracy = 0;
            for labels in Y:
                if Y[labels] == DT.label_prediction:
                    accuracy+=1
            return accuracy/len(Y)
        else: 
            Xleft = None
            Yleft = None
            Xright = None               #fixed
            Yright = None
            for sample in range(len(X)):
                if DT.feature_split_sign == "<":
                    if X[sample][DT.feature_split] >= DT.feature_split_value: 
                        if Xleft or Yleft is None:
                            Xleft = np.vstack([X[sample]])
                            Yleft = np.vstack([Y[sample]])
                        else:
                            Xleft = np.vstack([Xleft, X[sample]])
                            Yleft = np.vstack([Yleft, Y[sample]])
                    else:
                        if Xright or Yright is None:
                            Xright = np.vstack([X[sample]])
                            Yright = np.vstack([Y[sample]])
                        else:
                            Xright = np.vstack([Xright, X[sample]])
                            Yright = np.vstack([Yright, Y[sample]])
                else:
                    if X[sample][DT.feature_split] > DT.feature_split_value: 
                        if Xleft or Yleft is None:
                            Xleft = np.vstack([X[sample]])
                            Yleft = np.vstack([Y[sample]])
                        else:
                            Xleft = np.vstack([Xleft, X[sample]])
                            Yleft = np.vstack([Yleft, Y[sample]])
                    else:
                        if Xright or Yright is None:
                            Xright = np.vstack([X[sample]])
                            Yright = np.vstack([Y[sample]])
                        else:
                            Xright = np.vstack([Xright, X[sample]])
                            Yright = np.vstack([Yright, Y[sample]])

            return DT_test_real(Xleft, Yleft, DT.left_child) + DT_test_real(Xright, Yright, DT.right_child)

def DT_train_real_best(X_train = None, Y_train = None, X_val = None, Y_val = None):
    DT_best = DT_train_real(X_train, Y_train, -1)
    acc_best = DT_test_real(X_val, Y_val, DT_best)
    depth = maxDepth(DT_best)

    for x in range(depth):
        DT_curr = DT_train_real(X_train, Y_train, x)
        acc_curr = DT_test_real(X_val, Y_val, DT_curr)
        if acc_curr >= acc_best:
            acc_best = acc_curr
            DT_best = DT_curr

    return DT_best