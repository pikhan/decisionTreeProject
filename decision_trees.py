# Ibraheem Khan and Matthew Alighchi
# CS 491: Project 1 - Decision Trees
# Emily Hand - UNR

#This class will form the basis for our tree. Clearly, we need to denote
#left and right children of each node in the tree, but we also need to
#keep track of which samples are "in" each node during the model's training
#so that we can form output labels and see what is going on during the debugging
#process. Further, we need to make sure during training we do not split on
#the same feature twice so we will keep track of un-split features as well
#as which feature we choose to split on. These properties will all be None
#upon the return of DT_train_binary save for feature_split as the rest are
#only needed during training.

 #test commit
class Node(object):
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.samples = None
        self.feature_list = None
        self.feature_split = None

#This is a modular function that will calculate the sum of a list of arguments.
def sum(*X = None):
    if X is None:
        print('\n The sum array input is invalid \n')
    else:
        sum = 0
        i = 0
        for
def entropy(X = None, Y = None):

def information_gain():

def DT_train_binary(X = None, Y = None, max_depth = -2):
    if max_depth <= -2:
        print('\n Please enter a correct max_depth integer \n')
    if X is None:
        print('\n The X 2D Array is empty. \n')
    if Y is None:
        print('\n The Y 2D Array is empty. \n')



def DT_test_binary(X = None, Y = None, DT = None):

def DT_train_binary_best(X_train = None, Y_train = None, X_val = None, Y_val = None):

def DT_make_prediction(x = 0, DT = None):
    if x == 0:
        print('\n Please enter a correct sample index. \n')

def DT_train_real(X = None, Y = None, max_depth = -2):
    if max_depth <= -2:
        print('\n Please enter a correct max_depth integer \n')

def DT_test_real(X = None, Y = None, DT = None):

def DT_train_real_best(X_train = None, Y_train = None, X_val = None, Y_val = None):