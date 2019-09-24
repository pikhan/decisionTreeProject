# Ibraheem Khan and Matthew Alighchi
# CS 491: Project 1 - Decision Trees
# Emily Hand - UNR

#Numpy is imported for log functions

import numpy as np



#This class will form the basis for our tree. Clearly, we need to denote
#left and right children of each node in the tree, but we also need to
#keep track of which samples are "in" each node during the model's training
#so that we can form output labels and see what is going on during the debugging
#process. Further, we need to make sure during training we do not split on
#the same feature twice so we will keep track of un-split features as well
#as which feature we choose to split on. These properties will all be None
#upon the return of DT_train_binary save for feature_split as the rest are
#only needed during training.

class Node(object):
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.samples = None
        self.feature_list = None
        self.feature_split = None
        self.label = None



#This is a modular function that will calculate the sum of a list of arguments.

def sigma(*args):
    if args is None:
        print('\n The sum array input is invalid \n')
    else:
        summer = 0
        for arg in args:
            summer = summer + arg
        return summer



#This is a modular function that will calculate the entropy of the whole data set

def entropy(X = None, Y = None):
    for i in range(len(X)):



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
        if DT.left_child is None and DT.right_child is None:
            int accuracy = 0;
            for labels in Y:
                if Y[labels] == DT.label:
                    accuracy+1
            return (accuracy/range(Y))
        else:
            Xleft = np.array
            Yleft = np.array
            Xright = np.array
            Yright = np.array
            for sample in X:
                if X[sample][DT.feature_split] == 0:
                    Xleft.add(X[sample])
                    Yleft.add(Y[sample])
                else:
                    Xright.add(X[sample])
                    Yright.add(Y[sample])

            return DT_test_binary(Xleft, Yleft, DT.left_child) + DT_test_binary(Xright, Yright, DT.right_child)

def maxDepth(DT = None):
    if node is None: 
        return 0 ;  
    else : 
        lDepth = maxDepth(node.left) 
        rDepth = maxDepth(node.right) 
  
        # Use the larger one 
        if (lDepth > rDepth): 
            return lDepth+1
        else: 
            return rDepth+1

def DT_train_binary_best(X_train = None, Y_train = None, X_val = None, Y_val = None):
    DT_best = DT_train_binary(X_train, Y_train, -1)
    acc_best = DT_test_binary(X_val, Y_val, DT_best)
    int depth = maxDepth(DT_best)

    for x in range(depth):
        DT_curr = DT_train_binary(X_train, Y_train, x)
        acc_curr = DT_test_binary(X_val, Y_val, DT_curr)
        if acc_curr >= acc_best:
            acc_best = acc_curr
            DT_best = DT_curr

    return DT_best



def DT_make_prediction(x = 0, DT = None):
    if x == 0:
        print('\n Please enter a correct sample index. \n')

def DT_train_real(X = None, Y = None, max_depth = -2):
    if max_depth <= -2:
        print('\n Please enter a correct max_depth integer \n')

def DT_test_real(X = None, Y = None, DT = None):

def DT_train_real_best(X_train = None, Y_train = None, X_val = None, Y_val = None):