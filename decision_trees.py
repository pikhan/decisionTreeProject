# Ibraheem Khan and Matthew Alighchi
# CS 491: Project 1 - Decision Trees
# Emily Hand - UNR

#Numpy is imported for log functions

import numpy as np

class BinaryNode(object):
    def __init__(self, left_child, right_child, parent, samples, feature_list, feature_split, label_prediction):
        self.left_child = None #the left child of the node, another node
        self.right_child = None #the right child of the node, another node
        self.parent = None #the parent node
        self.samples = np.arange(5) #a 1D integer array of training sample indices "in" the node
        self.feature_list = np.arange(5) #a 1D integer array of features we have not yet split upon on the branch
        self.feature_split = 0 #an integer, the feature index for which we split on at the node
        self.label_prediction = 0 #an integer, 1 for yes, 0 for no the label prediction our DT will output at the particular node

class RealNode(object):
    def __init__(self, left_child, right_child, parent, samples, feature_list, feature_split, feature_split_value, feature_split_sign, label_prediction):
        self.left_child = None #the left child of the node, another node
        self.right_child = None #the right child of the node, another node
        self.parent = None #the parent node
        self.samples = np.arange(5) #a 1D integer array of training sample indices "in" the node
        self.feature_list = np.zeros(shape =(5,5)) #a 2D array of triples (feature_index, feature_value, feature_sign) for which we have not yet split upon on the branch
        self.feature_split = 0 #an integer, the feature index for which we split on at the node
        self.feature_split_value = 0 #a double, the value for which we split the feature on
        self.feature_split_sign = "less" #a string, less or leq resp. corresponding to either "<" or "<=", for which we split the feature value with
        self.label_prediction = 0 #an integer, 1 for yes, 0 for no the label prediction our DT will output at the particular node


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

def entropy(Y):
    if Y is None:
        print('\n The label set input is None, please correct. \n')
    count_yes = 0
    count_no = 0
    for i in range(len(Y)): #len(Y) corresponds to the number of rows, i.e., samples in the label set
        if Y[i][0] == 0: #if the i-th row at the 0-th column of the label set Y, that is, if the i-th sample's label is 0
            count_no += 1
        if Y[i][0] == 1:
            count_yes += 1
        else:
            print('\n Label Set has characters other than 0 or 1, please standardize notation. \n')
    prob_yes = count_yes/len(Y)
    prob_no = count_no/len(Y)
    H = -1*sigma(prob_yes*np.log2(prob_yes), prob_no*np.log2(prob_no)) #this is our total entropy
    return H



#This is a modular function that will compute the information gain between some current DT node and its children given the current node, the global Y 2D numpy array, and hypothetical left/right labels

def information_gain(DT, Y, leftLabels, rightLabels):
    currentLabels = np.empty(shape=(len(DT.samples),1)) #A 0-filled np array having as many rows as the samples in DT and only 1 column to create a label subset
    for i in range(len((DT.parent).samples)):
        currentLabels[i][0] = Y[DT.samples[i]][0] #this creates our label subset from the global Y-label subset
    H_current = entropy(currentLabels) #we first compute our entropies
    H_left = entropy(leftLabels)
    H_right = entropy(rightLabels)
    proportionLeft = len((DT.left_child).samples)/len(DT.samples) #add in the sizing coefficients in the formula for information gain
    proportionRight = len((DT.right_child).samples)/len(DT.samples)
    informationGain = H_current - proportionLeft*H_left - proportionRight*H_right #compute our information gain
    return informationGain



#This is a modular function that will compute the best possible feature_split the algorithm should make by maximizing the information gain in the binary case

def best_split_binary(DT, Y, X):
    info_gain_list = [] #this is a list of information gain values
    left_samples_list = [] #this will be a list of lists, where each internal list corresponds to the list of samples in the left for some i-th feature
    right_samples_list =[] #this will be a list of lists, where each internal list corresponds to the list of samples in the right for some i-th feature
    leftLabels = [] #this will be a list of lists, where each internal list corresponds to the list of labels for each sample in left_samples_list at each i-th feature
    rightLabels = []#this will be a list of lists, where each internal list corresponds to the list of labels for each sample in right_samples_list at each i-th feature
    for i in range(len(DT.feature_list)):
        left_samples_list.append([])
        right_samples_list.append([])
        leftLabels.append([])
        rightLabels.append([])
        for j in range(len(DT.samples)):
            if X[j][i] == 0:
                left_samples_list[i].append(j)
                leftLabels[i].append(Y[j][0])
            else:
                right_samples_list[i].append(j)
                rightLabels.append(Y[j][0])
    for k in range(len(DT.feature_list)):
        leftLabels_resized = np.empty(shape=(len(leftLabels[k]),1))
        for l in range(len(leftLabels[k])):
            leftLabels_resized[l][0] = leftLabels[k][l]
        rightLabels_resized = np.empty(shape=(len(rightLabels[k]),1))
        for l in range(len(rightLabels[k])):
            rightLabels_resized[l][0] = rightLabels[k][l]
        info_gain_list.append(information_gain(DT, Y, leftLabels_resized, rightLabels_resized))
    info_gain_array = np.asarray(info_gain_list)
    DT.feature_split = DT.feature_list[np.argmax(info_gain_array)]
    lSamples = np.asarray(left_samples_list[np.argmax(info_gain_array)])
    rSamples = np.asarray(right_samples_list[np.argmax(info_gain_array)])
    new_feature_list = np.delete(DT.feature_list, np.argmax(info_gain_array))
    adder = 0
    addertwo = 0
    for m in range(len(lSamples)):
        adder += Y[lSamples[m]][0]
    left_label_prediction = np.floor(adder/len(lSamples))
    for n in range(len(rSamples)):
        addertwo += Y[rSamples[m]][0]
    right_label_prediction = np.floor(adder/len(rSamples))
    left_child = BinaryNode(None, None, DT, lSamples, new_feature_list, None, left_label_prediction)
    right_child = BinaryNode(None, None, DT, rSamples, new_feature_list, None, right_label_prediction)
    DT.left_child = left_child
    DT.right_child = right_child




#This is a modular function that will compute the best possible feature_split the algorithm should make by maximizing the information gain in the real case

def best_split_real(DT, Y):
    print('Test function')


def recursor(DT, Y, X, iter = 0):
    if iter == 0:
        return 0
    else:
        best_split_binary(DT, Y, X)
        return recursor(DT.left_child, Y, X, (iter - 1)) + recursor(DT.right_child, Y, X, (iter - 1))



def recursor_indefinite(DT, Y, X):
    if (DT.feature_list).size == 0:
        return 0
    best_split_binary(DT, Y, X)
    return recursor_indefinite(DT.left_child, Y, X) + recursor(DT.right_child, Y, X)


def DT_train_binary(X, Y, max_depth = -1):
    if max_depth <= -2:
        print('\n Please enter a correct max_depth integer \n')
    if X is None:
        print('\n The X 2D Array is empty. \n')
    if Y is None:
        print('\n The Y 2D Array is empty. \n')
    adder = 0
    for i in range(len(Y)):
        adder += Y[i][0]
    label_prediction = np.floor(adder/len(Y))
    root = BinaryNode(None, None, None, np.arange(len(Y)), np.arange(len(X[0])), None, label_prediction)
    if max_depth == -1:
        recursor_indefinite(root, Y, X)
        return root
    if max_depth == 0:
        return root
    if max_depth > 0:
        recursor(root, Y, X, max_depth)
        return root


def DT_test_binary(X, Y, DT):
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

def maxDepth(DT):
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

def DT_train_binary_best(X_train, Y_train, X_val, Y_val):
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



def DT_make_prediction(x, DT): #assumes DT is binary (I believe this is correct)
    if (DT.left_child is None) and (DT.right_child is None):
        return DT.label_prediction
    if x is None:
        print('\n Please enter a correct sample index. \n')
    if x[DT.feature_split] == 0:
        return DT_make_prediction(x, DT.left_child)
    else:
        return DT_make_prediction(x, DT.right_child)

def DT_train_real(X, Y, max_depth = -2):
    if max_depth <= -2:
        print('\n Please enter a correct max_depth integer \n')

def DT_test_real(X, Y, DT):
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

def DT_train_real_best(X_train, Y_train, X_val, Y_val):
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