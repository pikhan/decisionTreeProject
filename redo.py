# Ibraheem Khan and Matthew Alighchi
# CS 491: Project 1 - Decision Trees
# Emily Hand - UNR

#Numpy is imported for log functions

import numpy as np

class BinaryNode(object):
    def __init__(self):
        self.left_child = None #the left child of the node, another node
        self.right_child = None #the right child of the node, another node
        self.parent = None #the parent node
        self.samples = np.arange(1) #a 1D integer array of training sample indices "in" the node
        self.feature_list = np.arange(1) #a 1D integer array of features we have not yet split upon on the branch
        self.feature_split = 0 #an integer, the feature index for which we split on at the node
        self.label_prediction = 0.0 #an integer, 1 for yes, 0 for no the label prediction our DT will output at the particular node

class RealNode(object):
    def __init__(self):
        self.left_child = None #the left child of the node, another node
        self.right_child = None #the right child of the node, another node
        self.parent = None #the parent node
        self.samples = np.arange(5) #a 1D integer array of training sample indices "in" the node
        self.feature_list = np.zeros(shape =(5,5)) #a 2D array of triples (feature_index, feature_value, feature_sign) for which we have not yet split upon on the branch
        self.feature_split = 0 #an integer, the feature index for which we split on at the node
        self.feature_split_value = 0 #a double, the value for which we split the feature on
        self.feature_split_sign = "less" #a string, less or leq resp. corresponding to either "<" or "<=", for which we split the feature value with
        self.label_prediction = 0.0 #an integer, 1 for yes, 0 for no the label prediction our DT will output at the particular node

def sigma(*args):
    if args is None:
        return 0
    else:
        summer = 0
        for arg in args:
            summer = summer + arg
        return summer



#This is a modular function that will calculate the entropy of the whole data set

def entropy(Y):
    if Y is None:
        return 0
    if len(Y) == 0:
        return 0
    count_yes = 0.0
    count_no = 0.0
    for i in range(len(Y)): #len(Y) corresponds to the number of rows, i.e., samples in the label set        
        if Y[i][0] == 0: #if the i-th row at the 0-th column of the label set Y, that is, if the i-th sample's label is 0
            count_no += 1
        else:
            count_yes += 1
    prob_yes = 0.0 
    prob_yes = count_yes/len(Y)
    prob_no = 0.0
    prob_no = count_no/len(Y)
    if (prob_yes == 0) and (prob_no != 0):
        H = -1*prob_no*np.log2(prob_no)
    else:
        if (prob_no == 0) and (prob_yes != 0):
            H = -1*prob_yes*np.log2(prob_yes)
        else:
            if (prob_no == 0) and (prob_yes == 0):
                H = 0
            else:
                H = (-1*prob_yes*np.log2(prob_yes)) - (prob_no*np.log2(prob_no)) #this is our total entropy
    return H



#This is a modular function that will compute the information gain between some current DT node and its children given the current node, the global Y 2D numpy array, and hypothetical left/right labels

def information_gain(Y, leftLabels, rightLabels):
    H_current = entropy(Y) #we first compute our entropies
    H_left = entropy(leftLabels)
    H_right = entropy(rightLabels)

    proportionLeft = 0.0
    proportionRight = 0.0
    if leftLabels is None:
        proportionLeft = 0.0
    else:
        proportionLeft = len(leftLabels)/len(Y) #add in the sizing coefficients in the formula for information gain
    if rightLabels is None:
        proportionRight = 0.0
    else:
        proportionRight = len(rightLabels)/len(Y)
    informationGain = H_current - proportionLeft*H_left - proportionRight*H_right #compute our information gain

    return informationGain

def DT_train_binary(X, Y, max_depth = -1):
    root = None
    if X is None: 
        return root
    root = BinaryNode()
    root.samples = X
    root.feature_list = np.arange(len(X[0]), dtype=int)
     
    left_IG = None
    right_IG = None

    best_IG = -1.0
    curr_IG = 0.0
    best_index = 0
    avg_yes = 0.0
    for feature in range(len(X[0])):
        avg_yes_curr = 0.0
        if (feature in root.feature_list) == True:
            for sample in range(len(X)):
                if X[sample][feature] == 0:
                    if left_IG is None:
                        left_IG = np.vstack([Y[sample]])
                    else:
                        left_IG = np.vstack([left_IG, Y[sample]])
                else:
                    avg_yes_curr += 1
                    if right_IG is None:
                        right_IG = np.vstack([Y[sample]])
                    else:
                        right_IG = np.vstack([right_IG, Y[sample]])
            
            curr_IG = information_gain(Y, left_IG, right_IG)
            left_IG = None
            right_IG = None
            if curr_IG > best_IG:
                avg_yes = avg_yes_curr
                best_IG = curr_IG
                best_index = feature

    for label in range(len(Y)):
        if Y[label][0] == 1:
            avg_yes +=1

    root.label_prediction = avg_yes / len(Y)
    if root.label_prediction < 0.5:
        root.label_prediction = 0
    else:
        root.label_prediction = 1

    if best_IG == -1.0:
        return root

    root.feature_split = best_index
    if max_depth == 0:
        return root
    max_depth -= 1

    left_child = BinaryNode()
    left_child.parent = root
    right_child = BinaryNode()
    right_child.parent = root

    root.left_child = left_child
    root.right_child = right_child
    
    Xleft = None
    Xright = None
    Yleft = None
    Yright = None

    for sample in range(len(X)):
        if X[sample][best_index] == 0:
            if Xleft is None:
                Xleft = np.vstack([X[sample]])
                Yleft = np.vstack([Y[sample]])
            else:
                Xleft = np.vstack([Xleft, X[sample]])
                Yleft = np.vstack([Yleft, Y[sample]])
        else:
            if Xright is None:
                Xright = np.vstack([X[sample]])
                Yright = np.vstack([Y[sample]])
            else:
                Xright = np.vstack([Xright, X[sample]])
                Yright = np.vstack([Yright, Y[sample]])
    remove_index = 0
    for i in range(len(root.feature_list)):
        if root.feature_list[i] == best_index:
            remove_index = i
    feature_list = np.delete(root.feature_list, remove_index)
    if Xleft is not None:
        DT_train_binary_aux(left_child, Xleft, Yleft, feature_list, max_depth)
    if Xright is not None:
        DT_train_binary_aux(right_child, Xright, Yright, feature_list, max_depth)

    return root

def DT_train_binary_aux(root, X, Y, feature_list, max_depth = -1):
    if X is None: 
        return root
    
    root.samples = X
    root.feature_list = feature_list
    
    left_IG = None
    right_IG = None

    best_IG = -1.0
    curr_IG = 0.0
    best_index = 0
    avg_yes = 0.0
    
    for feature in range(len(X[0])):
        avg_yes_curr = 0.0
        if (feature in root.feature_list) == True:
            for sample in range(len(X)):
                if X[sample][feature] == 0:
                    if left_IG is None:
                        left_IG = np.vstack([Y[sample]])
                    else:
                        left_IG = np.vstack([left_IG, Y[sample]])
                else:
                    if right_IG is None:
                        right_IG = np.vstack([Y[sample]])
                    else:
                        right_IG = np.vstack([right_IG, Y[sample]])
            curr_IG = information_gain(Y, left_IG, right_IG)
            left_IG = None
            right_IG = None
            if curr_IG > best_IG:
                avg_yes = avg_yes_curr
                best_IG = curr_IG
                best_index = feature
    #print(best_IG)
    for label in range(len(Y)):
        if Y[label][0] == 1:
            avg_yes +=1
    root.label_prediction = avg_yes / len(Y)
    if root.label_prediction < 0.5:
        root.label_prediction = 0
    else:
        root.label_prediction = 1

    if best_IG == -1.0:
        root = None
        return root


    root.feature_split = best_index


    if max_depth == 0:
        return root
    max_depth -= 1

    left_child = BinaryNode()
    left_child.parent = root
    right_child = BinaryNode()
    right_child.parent = root

    root.left_child = left_child
    root.right_child = right_child
    
    Xleft = None
    Xright = None
    Yleft = None
    Yright = None

    for sample in range(len(X)):
        if X[sample][best_index] == 0:
            if Xleft is None:
                Xleft = np.vstack([X[sample]])
                Yleft = np.vstack([X[sample]])
            else:
                Xleft = np.vstack([Xleft, X[sample]])
                Yleft = np.vstack([Yleft, Y[sample]])
        else:
            if Xright is None:
                Xright = np.vstack([X[sample]])
                Yright = np.vstack([Y[sample]])
            else:
                Xright = np.vstack([Xright, X[sample]])
                Yright = np.vstack([Yright, Y[sample]])
    if len(feature_list) == 0: 
        return root
    
    remove_index = 0
    for i in range(len(feature_list)):
        if feature_list[i] == best_index:
            remove_index = i
    feature_list = np.delete(root.feature_list, remove_index)

    if Xleft is not None:
        DT_train_binary_aux(left_child, Xleft, Yleft, feature_list, max_depth)
    if Xright is not None:
        DT_train_binary_aux(right_child, Xright, Yright, feature_list, max_depth)

    return root
def DT_test_binary(X, Y, DT):
    return DT_test_binary_aux(X, Y, DT) / len(DT.samples)

def DT_test_binary_aux(X, Y, DT):
    if (X is None) or (Y is None):
        return 0
    if DT is not None:
        if (DT.left_child is None) and (DT.right_child is None):
            accuracy = 0.0
            for labels in range(len(Y)):
                if Y[labels][0] == DT.label_prediction:
                    accuracy+=1
            if DT.parent is None:
                return accuracy
            else:
                return accuracy
        else:
            Xleft = None
            Yleft = None                 #fixed
            Xright = None
            Yright = None
            for sample in range(len(X)):
                if X[sample][DT.feature_split] == 0:
                    if Xleft is None:
                        Xleft = np.vstack([X[sample]])
                        Yleft = np.vstack([Y[sample]])
                    else:
                        Xleft = np.vstack([Xleft, X[sample]])
                        Yleft = np.vstack([Yleft, Y[sample]])
                else:
                    if Xright is None:
                        Xright = np.vstack([X[sample]])
                        Yright = np.vstack([Y[sample]])
                    else:
                        Xright = np.vstack([Xright, X[sample]])
                        Yright = np.vstack([Yright, Y[sample]])

            acc_total = DT_test_binary_aux(Xleft, Yleft, DT.left_child) + DT_test_binary_aux(Xright, Yright, DT.right_child)
            return acc_total 
    else:
        return 0.0

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
    depth = maxDepth(DT_best) - 1 
    for x in range(depth):
        DT_curr = DT_train_binary(X_train, Y_train, x)
        acc_curr = DT_test_binary(X_val, Y_val, DT_curr)
        #print(acc_curr)
        if acc_curr >= acc_best:
            acc_best = acc_curr
            DT_best = DT_curr

    return DT_best



def DT_make_prediction(x, DT): #assumes DT is binary (I believe this is correct)
    if (DT.left_child is None) and (DT.right_child is None):
        return DT.label_prediction
    if x[DT.feature_split] == 0:
        return DT_make_prediction(x, DT.left_child)
    else:
        return DT_make_prediction(x, DT.right_child)


def DT_train_real(X, Y, max_depth = -1):
    root = None
    if X is None: 
        return root
    root = RealNode()
    root.samples = X
    root.feature_list = np.arange(len(X[0]), dtype=int)
     
    left_IG = None
    right_IG = None

    best_IG = -1.0
    curr_IG = 0.0
    best_index = 0
    avg_yes = 0.0
    for feature in range(len(X[0]) -1):
        avg_yes_curr = 0.0
        if (feature in root.feature_list) == True:
            for sample in range(len(X)):
                if X[sample][feature] > X[sample][feature+1]:
                    if left_IG is None:
                        left_IG = np.vstack([Y[sample]])
                    else:
                        left_IG = np.vstack([left_IG, Y[sample]])
                else:
                    avg_yes_curr += 1
                    if right_IG is None:
                        right_IG = np.vstack([Y[sample]])
                    else:
                        right_IG = np.vstack([right_IG, Y[sample]])
            
            curr_IG = information_gain(Y, left_IG, right_IG)
            left_IG = None
            right_IG = None
            if curr_IG > best_IG:
                avg_yes = avg_yes_curr
                best_IG = curr_IG
                best_index = feature

    for label in range(len(Y)):
        if Y[label][0] == 1:
            avg_yes +=1

    root.label_prediction = avg_yes / len(Y)
    if root.label_prediction < 0.5:
        root.label_prediction = 0
    else:
        root.label_prediction = 1

    if best_IG == -1.0:
        return root

    root.feature_split = best_index
    if max_depth == 0:
        return root
    max_depth -= 1

    left_child = RealNode()
    left_child.parent = root
    right_child = RealNode()
    right_child.parent = root

    root.left_child = left_child
    root.right_child = right_child
    
    Xleft = None
    Xright = None
    Yleft = None
    Yright = None

    for sample in range(len(X) - 1):
        if X[sample][best_index] > X[sample+1][best_index]:
            if Xleft is None:
                Xleft = np.vstack([X[sample]])
                Yleft = np.vstack([Y[sample]])
            else:
                Xleft = np.vstack([Xleft, X[sample]])
                Yleft = np.vstack([Yleft, Y[sample]])
        else:
            if Xright is None:
                Xright = np.vstack([X[sample]])
                Yright = np.vstack([Y[sample]])
            else:
                Xright = np.vstack([Xright, X[sample]])
                Yright = np.vstack([Yright, Y[sample]])
    remove_index = 0
    for i in range(len(root.feature_list)):
        if root.feature_list[i] == best_index:
            remove_index = i
    feature_list = np.delete(root.feature_list, remove_index)
    if Xleft is not None:
        DT_train_real_aux(left_child, Xleft, Yleft, feature_list, max_depth)
    if Xright is not None:
        DT_train_real_aux(right_child, Xright, Yright, feature_list, max_depth)

    return root

def DT_train_real_aux(root, X, Y, feature_list, max_depth = -1):
    if X is None: 
        return root
    
    root.samples = X
    root.feature_list = feature_list
    
    left_IG = None
    right_IG = None

    best_IG = -1.0
    curr_IG = 0.0
    best_index = 0
    avg_yes = 0.0
    
    for feature in range(len(X[0]) - 1):
        avg_yes_curr = 0.0
        if (feature in root.feature_list) == True:
            for sample in range(len(X) - 1):
                if X[sample][feature] > X[sample+1][best_index]:
                    if left_IG is None:
                        left_IG = np.vstack([Y[sample]])
                    else:
                        left_IG = np.vstack([left_IG, Y[sample]])
                else:
                    if right_IG is None:
                        right_IG = np.vstack([Y[sample]])
                    else:
                        right_IG = np.vstack([right_IG, Y[sample]])
            curr_IG = information_gain(Y, left_IG, right_IG)
            left_IG = None
            right_IG = None
            if curr_IG > best_IG:
                avg_yes = avg_yes_curr
                best_IG = curr_IG
                best_index = feature
    #print(best_IG)
    for label in range(len(Y)):
        if Y[label][0] == 1:
            avg_yes +=1
    root.label_prediction = avg_yes / len(Y)
    if root.label_prediction < 0.5:
        root.label_prediction = 0
    else:
        root.label_prediction = 1

    if best_IG == -1.0:
        root = None
        return root


    root.feature_split = best_index


    if max_depth == 0:
        return root
    max_depth -= 1

    left_child = RealNode()
    left_child.parent = root
    right_child = RealNode()
    right_child.parent = root

    root.left_child = left_child
    root.right_child = right_child
    
    Xleft = None
    Xright = None
    Yleft = None
    Yright = None

    for sample in range(len(X) - 1):
        if X[sample][best_index] > X[sample+1][best_index]:
            if Xleft is None:
                Xleft = np.vstack([X[sample]])
                Yleft = np.vstack([X[sample]])
            else:
                Xleft = np.vstack([Xleft, X[sample]])
                Yleft = np.vstack([Yleft, Y[sample]])
        else:
            if Xright is None:
                Xright = np.vstack([X[sample]])
                Yright = np.vstack([Y[sample]])
            else:
                Xright = np.vstack([Xright, X[sample]])
                Yright = np.vstack([Yright, Y[sample]])
    if len(feature_list) == 0: 
        return root
    
    remove_index = 0
    for i in range(len(feature_list)):
        if feature_list[i] == best_index:
            remove_index = i
    feature_list = np.delete(root.feature_list, remove_index)

    if Xleft is not None:
        DT_train_real_aux(left_child, Xleft, Yleft, feature_list, max_depth)
    if Xright is not None:
        DT_train_real_aux(right_child, Xright, Yright, feature_list, max_depth)

    return root

def DT_test_real(X, Y, DT):
    return DT_test_real_aux(X, Y, DT)/ len(Y)
def DT_test_real_aux(X, Y, DT):
    if (X is None) or (Y is None):
        return 0
    if DT is not None:
        if (DT.left_child is None) and (DT.right_child) is None:
            accuracy = 0.0;
            for labels in range(len(Y)):
                if Y[labels][0] == DT.label_prediction:
                    accuracy+=1
            if DT.parent is None:
                return accuracy
            else:
                return accuracy
        else: 
            Xleft = None
            Yleft = None
            Xright = None               #fixed
            Yright = None
            for sample in range(len(X)):
                if X[sample][DT.feature_split] > DT.feature_split_value: 
                    if Xleft is None:
                        Xleft = np.vstack([X[sample]])
                        Yleft = np.vstack([Y[sample]])
                    else:
                        Xleft = np.vstack([Xleft, X[sample]])
                        Yleft = np.vstack([Yleft, Y[sample]])
                else:
                    if Xright is None:
                        Xright = np.vstack([X[sample]])
                        Yright = np.vstack([Y[sample]])
                    else:
                        Xright = np.vstack([Xright, X[sample]])
                        Yright = np.vstack([Yright, Y[sample]])

            acc_total = DT_test_real_aux(Xleft, Yleft, DT.left_child) + DT_test_real_aux(Xright, Yright, DT.right_child)
            return acc_total

def DT_train_real_best(X_train, Y_train, X_val, Y_val):
    DT_best = DT_train_real(X_train, Y_train, -1)
    acc_best = DT_test_real(X_val, Y_val, DT_best)
    depth = maxDepth(DT_best) - 1 
    for x in range(depth):
        DT_curr = DT_train_real(X_train, Y_train, x)
        acc_curr = DT_test_real(X_val, Y_val, DT_curr)
        #print(acc_curr)
        if acc_curr >= acc_best:
            acc_best = acc_curr
            DT_best = DT_curr

    return DT_best