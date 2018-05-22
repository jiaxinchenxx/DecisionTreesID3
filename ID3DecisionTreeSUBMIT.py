#coding:UTF-8


from scipy.io import loadmat, savemat
import numpy as np
from random import choice
from sklearn.ensemble import  RandomForestClassifier
import pickle

from sklearn.model_selection import KFold
from numpy import *

#from graphviz import Digraph


#dot = Digraph()  # this is for visualization, graphviz environment needed, if there's not, please comment this and the
#counter = 1      # visualization part, this counter is the identity of tree node for visualisation

class tree:

    def __init__(self,op, cl,wil):  # op = -1 this is a leaf node
        self.op = op            # cl = -1 this is a middle node
        self.cl = cl            # identity is used for visualization
        self.kids = []
        self.identity = -1      # kids are a list of size 2
        self.wil = wil          # wil for confidence factor

    def set_op(self, op):
        self.op = op

    def set_kids(self, nkid, pkid):
        self.kids.append(nkid)
        self.kids.append(pkid)

    def set_class(self, cl):
        self.cl = cl

def Creating_Decision_Tree(labels, attrs, examples):

    p_num = len(np.where(labels == 1)[0])  # positive targets
    n_num = len(np.where(labels != 1)[0])  # negative targets
    num = float(len(labels))               # total targets

    if p_num == num:                   # if all targets are the same
        wil = WilsonSI(p_num, 0)       # Wilson Score Interval, details are in the report
        node = tree(-1,1,wil)
        return node
    if n_num == num:
        wil = WilsonSI(n_num, 0)
        node = tree(-1,0,wil)
        return node
    if float(p_num) / num < 0.02:     # if there is only very little examples of pos or neg
        wil = WilsonSI(n_num, p_num)
        node = tree(-1,0,wil)
        return node
    if float(p_num) / num > 0.98:
        wil = WilsonSI(p_num, n_num)
        node = tree(-1,1,wil)
        return node

    if len(attrs) == 0:               # if there's no attrs left
        if p_num > n_num:
            wil = WilsonSI(p_num, n_num)  # Wilson Score Interval, details are in the report
            node = tree(-1, 1, wil)
            return node
        else:
            wil = WilsonSI(n_num, p_num)
            node = tree(-1, 0, wil)
            return node

    E_D = -p_num / num * np.log2(p_num / num) - n_num / num * np.log2(n_num / num)  # Entropy of the targets
    # calculation of E(D)

    max_inx = 0 # max index
    max_IG = 0.0 # max value

    IG_Array = np.zeros(45) # in case there are multiple max values
    IG_Array[:] = -1 # there are situations that Entropy is 0

    for i in attrs:

        p_attrinx_1 = np.where(examples[:,i] == 1)[0]
        n_attrinx_0 = np.where(examples[:,i] == 0)[0]
        num_attr = float(len(p_attrinx_1) + len(n_attrinx_0))
        # find inx of attributes with 1 & 0 respectively
        labels_inx_1 = labels[p_attrinx_1]
        labels_inx_0 = labels[n_attrinx_0]
        # find their corresponding labels / targets
        p_labels_1 = len(np.where(labels_inx_1 == 1)[0])
        n_labels_1 = len(np.where(labels_inx_1 != 1)[0])
        num_labels_1 = float(len(labels_inx_1))
        # find the amount of pos & neg of labels with attr 1
        if p_labels_1 == 0 or n_labels_1 == 0: # if there are all pos or neg
            E_attr_1 = 0
        else:
            E_attr_1 = -p_labels_1 / num_labels_1 * np.log2(p_labels_1 / num_labels_1) - n_labels_1 / num_labels_1 * np.log2(n_labels_1 / num_labels_1)
        # E(1)


        p_labels_0 = len(np.where(labels_inx_0 == 1)[0])
        n_labels_0 = len(np.where(labels_inx_0 != 1)[0])
        num_labels_0 = float(len(labels_inx_0))
        # find pos & neg of labels with attr 0
        if p_labels_0 == 0 or n_labels_0 == 0: # if there are all pos neg
            E_attr_0 = 0
        else:
            E_attr_0 = -p_labels_0 / num_labels_0 * np.log2(p_labels_0 / num_labels_0) - n_labels_0 / num_labels_0 * np.log2(n_labels_0 / num_labels_0)
        # E(0)


        p = len(p_attrinx_1) / num_attr
        n = len(n_attrinx_0) / num_attr
        IG = E_D - p * E_attr_1 - n * E_attr_0 # Gnformation Gain
        IG_Array[i] = IG # store the information gain

        if IG > max_IG: # find the maximum IG
            max_IG = IG

    MAX_INX = np.where(IG_Array == max_IG)[0] # in case there's multiple maximum

    if len(MAX_INX) > 1:
        max_inx = choice(MAX_INX) # randomly select attribute while the IG are the same
    if len(MAX_INX) == 1:
        max_inx = MAX_INX[0]



    targets_inx_1 = np.where(examples[:, max_inx] == 1)[0] # in terms of the best_attribute, divide the targets
    targets_inx_0 = np.where(examples[:, max_inx] == 0)[0] # same

    targets_1 = np.array(labels[targets_inx_1]) # targets with best_attr = 1
    targets_0 = np.array(labels[targets_inx_0]) # targets with best_attr = 0

    examples_1 = np.array(examples[targets_inx_1, :]) # examples with best_attr = 1
    examples_0 = np.array(examples[targets_inx_0, :]) # examples with best_attr = 0


    if examples_0.shape[0] == 0 or examples_1.shape[0] == 0: # if any new example is empty
        if p_num > n_num:                                    # MAJORITY CASE
            wil = WilsonSI(p_num, n_num)
            node = tree(-1, 1,wil)                               # make classification
            return node                                      # RETURN
        else:
            wil = WilsonSI(n_num, p_num)
            node = tree(-1, 0,wil)
            return node


    node = tree(max_inx,-1,0.0)   # otherwise, here is a middle node with op = max_inx, cl = -1 means no leaf
    attrs.remove(max_inx)     # this best_attr won't be considered again

    attrs_new_0 = list(attrs) # new attr list
    attrs_new_1 = list(attrs) # new attr list

    n_kid = Creating_Decision_Tree(targets_0, attrs_new_0, examples_0) # recursively generate nodes
    p_kid = Creating_Decision_Tree(targets_1, attrs_new_1, examples_1) # same

    node.set_kids(n_kid, p_kid) # get the kids

    return node # return itself to its parent


def Select_labels(i,Y_TRAIN):          # select which emotion you want to classify 1-6 can be selected
    labels = np.array(Y_TRAIN)
    labels[np.where(Y_TRAIN ==i)] = 1
    labels[np.where(Y_TRAIN !=i)] = 0
    return labels

def evaluation(root, exa):  # naive evaluation given root and example
    cl = -1
    if root.op != -1: # if not leaf node
        if exa[root.op] == 0: # if negative case
            cl = evaluation(root.kids[0], exa) # from neg child get the classification
            return cl
        else:
            cl = evaluation(root.kids[1], exa) # if positive case
            return cl                          # from pos child get the classification
    if root.op == -1:    # if leaf node
        return root.cl   # return classification

    return cl

def evaluation_with_wil(root, exa):  # evaluation considered the Wilson Score Interval
    cl = -1
    wil = 0.0
    if root.op != -1: # if not leaf node
        if exa[root.op] == 0: # if negative case
            cl, wil = evaluation_with_wil(root.kids[0], exa) # from neg child get the classification
            return cl, wil
        else:
            cl, wil = evaluation_with_wil(root.kids[1], exa) # if positive case
            return cl, wil                          # from pos child get the classification
    if root.op == -1:    # if leaf node
        return root.cl,root.wil   # return classification

    return cl, wil

def evaluation_with_trace(root, exa):  # evaluation considered the depth of looking through the classification
    cl = -1
    c = 0
    if root.op != -1: # if not leaf node
        if exa[root.op] == 0: # if negative case
            cl,c= evaluation_with_trace(root.kids[0], exa) # from neg child get the classification
            return cl, 1 + c
        else:
            cl,c= evaluation_with_trace(root.kids[1], exa) # if positive case
            return cl, 1 + c                          # from pos child get the classification
    if root.op == -1:    # if leaf node
        return root.cl,1   # return classification

    return cl, c

def WilsonSI(majority, less): # method to compute the WSI of the majority-value
    N = majority + less   # amount of the examples
    WilsonConfidence = 0.0
    WilsonConfidence += (majority + 1.96**2 / 2) / (N +  1.96**2) # pure math, details are in report
    WilsonConfidence -= (1.96 / (N + 1.96**2) ) * sqrt((majority * less / N) + (1.96 ** 2 / 4.0))
    return WilsonConfidence

def EVALUATION(Root_List, exam): # naive PREDICTION
    res = []
    for i in range(6):
        res.append(evaluation(Root_List[i], exam)) # get the evaluation from each root
    res = np.array(res)
    inx = np.where(res == 1)[0]

    if len(inx) == 0:
        return choice([T for T in range(1,7)])
    else:
        return choice(inx) + 1 # randomly return one of the multiple positive emotion


def EVALUATION4TR(ROOT_List, exam): # evaluation considered depth
    res = []
    for i in range(6):
        res.append(evaluation_with_trace(ROOT_List[i], exam))  # get the evaluation from each root
    res = np.array(res)
    inx = np.where(res == 1)[0]
    #print inx+1
    if len(inx) == 0:
        M = choice([T for T in range(1,7)])
        #print M
        return M
    else:
        min_tr = np.min(res[inx, 1]) # find the minimal trace of multiple postive emotions
        pos = np.where(res[inx,1] == min_tr)[0]
        T = choice(inx[pos])  # return the corresponding emotion of relatively small trace
        print T+1
        return T + 1

def EVALUATION4WIL(ROOT_List, exam): # evaluation considered WIS
    res = []
    for i in range(6):
        res.append(evaluation_with_wil(ROOT_List[i], exam))  # get the evaluation from each root

    res = np.array(res)
    inx = np.where(res == 1)[0]

    if len(inx) == 0:
        T = np.where(res[:,1] == np.min(res[:,1]))[0] # find the least confident one
        return T[0] + 1
    elif len(inx) == 1:   # return
        return inx[0] + 1
    else:
        max_wil = np.max(res[inx, 1])  # find the most confident one
        pos = np.where(res[inx, 1] == max_wil)[0]
        T = choice(inx[pos])
        return T + 1


def OneVersusOneTraining(X_TRAIN, ATTR, Y_TRAIN):  # This function is for One Versus. One Traning
    ROOT_DIC = {}
    cl_inx = []
    for i in range(1,7):
        inx = np.where(Y_TRAIN == i)[0]  # get the inx of every single emotion
        cl_inx.append(inx)
    for i in range(1,7):
        for j in range(i + 1,7):
            cl0 = cl_inx[i - 1]    # select two emotions to train
            cl1 = cl_inx[j - 1]
            TARGETS_0 = np.array(Y_TRAIN[cl0])
            TARGETS_0[:] = 0   # the former one is viewed as negative, later one is negtive
            TARGETS_1 = np.array(Y_TRAIN[cl1])
            TARGETS_1[:] = 1
            TARGETS = np.append(TARGETS_0, TARGETS_1, axis = 0)   # Get targets of the two specific emotions
            EXAMPLES = np.append(X_TRAIN[cl0], X_TRAIN[cl1], axis= 0) # Select out corresponding examples


            ATTRIBUTES = FeatureSelection(EXAMPLES, TARGETS) # Select the most powerful feature

            root = Creating_Decision_Tree(TARGETS, ATTRIBUTES, EXAMPLES) # get the classifier

            ROOT_DIC['{m}Versus{n}'.format(m = i, n = j)] = root
    return ROOT_DIC


def FeatureSelection(X_TRAIN, Y_TRAIN):

    VOTE = np.zeros(45)
    for i in range(10):
        rf = RandomForestClassifier()
        rf.fit(X_TRAIN, Y_TRAIN.reshape(Y_TRAIN.shape[0]))
        rank = np.argsort(rf.feature_importances_)
        for j in range(38):
            VOTE[rank[j]] += 1

    VOTE = np.argsort(VOTE)

    ATTRIBUTES = []
    for i in range(7):
        ATTRIBUTES.append(VOTE[i])

    return ATTRIBUTES

def Vote4Selection(exam, ROOT_DIC, wil, f = None):
    res = np.zeros(6)
    for i in range(1, 7):
        for j in range(i + 1, 7):
            l = evaluation(ROOT_DIC['{m}Versus{n}'.format(m=i, n=j)], exam)
            if l == 0:
                res[i - 1] += 1
            if l == 1:
                res[j - 1] += 1
    T = np.where(res == np.max(res))[0]
    if len(T) > 1:
        if f == None:
            inx = np.where(wil[T, 1] == np.min(wil[T, 1]))[0]
            return T[inx] + 1
        else:
            max_inx = -1
            max = 0.0
            for i in range(6):
                if wil[i, 0] == 1 and i in T:
                    if wil[i, 1] > max:
                        max = wil[i, 1]
                        max_inx = i
            if max == 0.0:
                for i in range(6):
                    if wil[i, 0] == 1 and wil[i, 1] > max:
                        max = wil[i, 1]
                        max_inx = i
            return max_inx + 1
    return T + 1

def EVALUATION4OVSO(ROOT_List, ROOT_DIC, exam):
    res = []
    for i in range(6):
        res.append(evaluation_with_wil(ROOT_List[i], exam))  # get the evaluation from each root
    res = np.array(res)
    inx = np.where(res == 1)[0]
    if len(inx) == 0:
        T = Vote4Selection(exam, ROOT_DIC, res)
        return T[0]
    if len(inx) > 1:
        T = Vote4Selection(exam, ROOT_DIC, res, 1)
        return T
    if len(inx) == 1:
        return inx[0] + 1


def Accuracy(PREDICTION_LA, Y_TEST): # just naively test accuracy
    error_count = 0.0
    for i in range(Y_TEST.shape[0]):
        if PREDICTION_LA[i] != Y_TEST[i]:
            error_count += 1
    print 1 - error_count / Y_TEST.shape[0]
    return 1 - error_count / Y_TEST.shape[0]

def COMBINATION(X_TRAIN, ATTR, Y_TRAIN):  # Here is the combination evaluation method
    ROOT_List = []
    for i in range(1,7):         # for each emotion, generate its own targets
        Y_TRAIN_i = Select_labels(i, Y_TRAIN)
        ROOT_List.append(Creating_Decision_Tree(Y_TRAIN_i, list(ATTR), X_TRAIN)) # get root for each emotion
    return ROOT_List            # this method returns 6 trees with full attributes


def COMBINATIONWITHFEA(X_TRAIN, ATTR, Y_TRAIN):   # for each emotion, generate its own targets
    ROOT_List = []
    for i in range(1,7):
        Y_TRAIN_i = Select_labels(i, Y_TRAIN)  # for each emotion, generate its own targets
        ATTR = FeatureSelection(X_TRAIN, Y_TRAIN_i)
        ROOT_List.append(Creating_Decision_Tree(Y_TRAIN_i,ATTR, X_TRAIN))
    return ROOT_List  # this method returns 6 trees with selected attributes


def prediction(ROOT_List, X_TEST): #, ROOT_DIC = None):
    prediction_label = zeros(X_TEST.shape[0], int)
    for i in range(X_TEST.shape[0]):
        #prediction_label[i] = EVALUATION(ROOT_List, X_TEST[i])
        #prediction_label[i] = EVALUATION4TR(ROOT_List, X_TEST[i])
        #prediction_label[i] = EVALUATION4OVSO(ROOT_List, ROOT_DIC, X_TEST[i])
        prediction_label[i] = EVALUATION4WIL(ROOT_List, X_TEST[i])

    return prediction_label


def Confusion_matrix(prediction_label, test_label, no_attribute):
    confusion_matrix = zeros((no_attribute, no_attribute), float)
    for i in range(len(prediction_label)):
        a = prediction_label[i]
        b = test_label[i]
        confusion_matrix[a - 1][b - 1] += 1

    return confusion_matrix


def recall_precision(confusion_matrix):
    recall = zeros((confusion_matrix.shape[0]), float)
    precision = zeros((confusion_matrix.shape[0]), float)
    for i in range(len(confusion_matrix)):
        recall[i] = confusion_matrix[i][i] / sum(confusion_matrix[i])
        precision[i] = confusion_matrix[i][i] / sum(confusion_matrix[:, i])
    average_recall = sum(recall) / len(recall)
    average_precision = sum(precision) / len(precision)
    #     return recall,precision
    return average_recall, average_precision


def normalised_recall_precision(confusion_matrix):
    recall = zeros((confusion_matrix.shape[0]), float)
    precision = zeros((confusion_matrix.shape[0]), float)
    for i in range(len(confusion_matrix)):
        confusion_matrix[i] = confusion_matrix[i] / sum(confusion_matrix[i])

    for i in range(len(confusion_matrix)):
        recall[i] = confusion_matrix[i][i] / sum(confusion_matrix[i])
        precision[i] = confusion_matrix[i][i] / sum(confusion_matrix[:, i])

    average_recall = sum(recall) / len(recall)
    average_precision = sum(precision) / len(precision)
    #     return recall,precision
    return average_recall, average_precision


def F1_measure(recall, precision):
    return (2 * precision * recall) / (precision + recall)


def Visulization(root, parent = None):  # this part is for visualization
    global counter  # unique identity of each node
    if root.op != -1: # if this is a middle node
        dot.node(str(counter), str(root.op), shape = 'circle') # add a node to the tree
        root.identity = counter # give this tree an identity
        counter += 1 # indentity counter accumulate

        if parent != None: # if there is a parent of this node
            dot.edge(str(parent), str(root.identity))  # then add an edge between them

        Visulization(root.kids[0], root.identity) # visualize its child
        Visulization(root.kids[1], root.identity) # same

    if root.op == -1: # if this is a leaf node
        if root.cl == 0: # if this is a negative case
            dot.node(str(counter), 'NO', shape = 'box') # add a negative node
            root.identity = counter # give this tree an identity
            counter += 1                # indentity counter accumulate
            if parent != None:          # if it has a parent, add an edge
                dot.edge(str(parent), str(root.identity))
            return # this path has been visualized
        else:
            dot.node(str(counter), 'YES', shape = 'box')  # basically the same
            root.identity = counter
            counter += 1
            if parent != None:
                dot.edge(str(parent), str(root.identity))
            return
    return


def CROSS_VAL(TARGETS, EXAMPLES, ATTR):
    Error = 0.0
    kf = KFold(n_splits=10, shuffle=True)  # 10-fold
    ROOT_List = []
    confusion_matrix = zeros((6, 6), float)
    for train, test in kf.split([i for i in range(1004)]):  # 1004 examples
        #print ('--------------------------------------------------------------------------')
        #ROOT_List = COMBINATION(EXAMPLES[train], ATTR, TARGETS[train])  # get the root list to evaluate
        ROOT_List = COMBINATIONWITHFEA(EXAMPLES[train], ATTR, TARGETS[train]) # get the root list with trees of less attributes
        #ROOT_DIC = OneVersusOneTraining(EXAMPLES[train], ATTR, TARGETS[train])  # ROOT_DIC denotes the root of voting system

        #prediction_label = prediction(ROOT_List, EXAMPLES[test], TARGETS[test])
        prediction_label = prediction(ROOT_List, EXAMPLES[test]) #,ROOT_DIC)
        Error += Accuracy(prediction_label, TARGETS[test])
        confusion_matrix = confusion_matrix + Confusion_matrix(prediction_label, TARGETS[test], 6)

    Error = Error / 10.0

    print('Confusion_matrix:')
    print(confusion_matrix)
    print('Recall and Precision:')
    recall, precision = normalised_recall_precision(confusion_matrix)
    print(recall, precision)
    print('F1_Measures:')
    print(F1_measure(recall, precision))

    return (1.0 - Error) * 100

def testTrees_WILSON(T, x2):
    predictions = zeros(x2.shape[0])
    for i in range(x2.shape[0]):
        predictions[i] = EVALUATION4WIL(T, x2[i])
    return predictions


def testTrees(T, x2):
    predictions = zeros(x2.shape[0])
    for i in range(x2.shape[0]):
        predictions[i] = EVALUATION4OVSO(T, ROOT_DIC, x2[i])
    return predictions

if __name__ == '__main__':

    filename = 'ROOT_SUBMIT.pkl'

    ROOT = pickle.load(open(filename, 'rb'))

    T = ROOT[0]
    ROOT_DIC = ROOT[1]

    test_filename = '.mat' # please input the test file name here

    data = loadmat(test_filename)
    x2 = data['x']   # I suppose your test name is x
    targets = data['y'] # test target is y

    predictions = testTrees(T, x2)  # there is another method to predict the emotions given called testTrees_WILSON()
                                    # if you want to have a try, please uncomment the following code, maybe it will achieve
                                    # a higher performance, it depends on the situation

    #predictions = testTrees_WILSON(T, x2)  # uncomment here if you want to try.

    #########################################################################################
    # if you want to use my build in accuracy test function, please uncomment the following:#
    #acc = Accuracy(predictions, targets)
    #########################################################################################

