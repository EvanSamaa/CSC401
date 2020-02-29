import argparse
import os
import numpy as np
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier



def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    accuracy_value = 0
    total_count = C.sum()
    correct_count = 0
    for i in range(0, C.shape[0]):
        correct_count = correct_count + C[i, i]
    accuracy_value = float(correct_count/total_count)
    return accuracy_value


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    rt_list = [0, 0, 0, 0]
    for i in range(0, C.shape[0]):
        class_count = 0
        for j in range (0, C.shape[0]):
            class_count = class_count + C[i, j]
        rt_list[i] = C[i,i]/class_count

    return rt_list



def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    rt_list = [0, 0, 0, 0]
    for i in range(0, C.shape[0]):
        class_count = 0
        for j in range(0, C.shape[0]):
            class_count = class_count + C[j, i]
        rt_list[i] = C[i, i] / class_count
    return rt_list


def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
    Returns:
       i: int, the index of the supposed best classifier
    '''
    label_names = ["Left", "Center", "Right", "Alt"]

    # generate and train classifiers
    classfiers = [None, None, None, None, None]
    classfiers[0] = SGDClassifier()
    classfiers[0].fit(X_train, y_train)
    classfiers[1] = GaussianNB()
    classfiers[1].fit(X_train, y_train)
    classfiers[2] = RandomForestClassifier(max_depth=5, n_estimators=10)
    classfiers[2].fit(X_train, y_train)
    classfiers[3] = MLPClassifier(alpha=0.5)
    classfiers[3].fit(X_train, y_train)
    classfiers[4] = AdaBoostClassifier()
    classfiers[4].fit(X_train, y_train)
    # generate results
    confusion_matrices = [None, None, None, None, None]
    acc_recall_precision = [None, None, None, None, None]
    classfier_names = ["SGDClassifier", "GaussianNB", "RandomForestClassifier", "MLPClassifier", "AdaBoostClassifier"]
    iBest = 0
    for i in range (0, 5):
        y_pred = classfiers[i].predict(x_test)
        confusion_matrices[i] = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3 ])
        accuracy_val = accuracy(confusion_matrices[i])
        recall_val = recall(confusion_matrices[i])
        precision_val = precision(confusion_matrices[i])
        acc_recall_precision[i] = [accuracy_val, recall_val, precision_val]

        if accuracy_val > acc_recall_precision[iBest][0]:
            iBest = i

    # print('TODO Section 3.1')

    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        # For each classifier, compute results and write the following output:
        #
        for i in range (0, 5):
            outf.write(f'Results for {classfier_names[i]}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {acc_recall_precision[i][0]:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in acc_recall_precision[i][1]]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in acc_recall_precision[i][2]]}\n')
            outf.write(f'\tConfusion Matrix: \n{confusion_matrices[i]}\n\n')
    # print(iBest)
    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest = 4):
    ''' This function performs experiment 3.2
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''

    # initialize variables
    classfier_choices = [SGDClassifier, GaussianNB, RandomForestClassifier, MLPClassifier, AdaBoostClassifier]
    X_Nk = [None, None, None, None, None]
    y_Nk = [None, None, None, None, None]
    size_retained = [1/32, 5/32, 10/32, 15/32, 20/32]
    classfiers = [None, None, None, None, None]
    discarded_X = None
    discarded_y = None
    confusion_matrices = [None, None, None, None, None]
    acc_List = [None, None, None, None, None]

    for i in range (0, 5):
        discarded_X, X_Nk[i], discarded_y, y_Nk[i] = train_test_split(X_train, y_train, test_size=size_retained[i], random_state=0)
        classfiers[i] = classfier_choices[iBest]()
        classfiers[i].fit(X_Nk[i], y_Nk[i])
        y_pred = classfiers[i].predict(x_test)
        confusion_matrices[i] = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
        acc_List[i] = accuracy(confusion_matrices[i])



    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # For each number of training examples, compute results and write
        # the following output:
        #     outf.write(f'{num_train}: {accuracy:.4f}\n'))
        for i in range (0, 5):
            outf.write(f'{y_Nk[i].shape[0]}: {acc_List[i]:.4f}\n')
        pass
    return (X_Nk[0], y_Nk[0])


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    classfier_choices = [SGDClassifier, GaussianNB, RandomForestClassifier, MLPClassifier, AdaBoostClassifier]
    k_vals = [5, 50]
    pp_values_32k = [[], []]
    pp_values_1k = [[], []]
    X_reduced_32k = [None, None] # [(32k,5), (32k,50)]
    X_reduced_1k = [None, None] # for k = 5 vs k = 50 respectively
    X_reduced_test = [None, None] # for 32k and 1k respectively
    models_32k = [None, None]
    models_1k = [None, None]
    accuracies_32k = [0, 0]
    accuracies_1k = [0, 0]
    k5_1k_feats = []
    k5_32k_feats = []

    for k in range(0, len(k_vals)):
        k_val = k_vals[k]
        # use different selector to select for different datasets
        selector_32k = SelectKBest(score_func=f_classif, k = k_val)
        X_reduced_32k[k] = selector_32k.fit_transform(X_train, y_train)
        selector_1k = SelectKBest(score_func=f_classif, k=k_val)
        X_reduced_1k[k] = selector_1k.fit_transform(X_1k, y_1k)
        # select k largest p values for the 32k dataset
        pp_32k = np.nan_to_num(selector_32k.pvalues_, 0)
        arg_pp_32k = pp_32k.argsort()
        # do the same for 1k dataset
        pp_1k = np.nan_to_num(selector_1k.pvalues_, 0)
        arg_pp_1k = pp_1k.argsort()
        # record the p_values and the index
        pp_values_32k[k] = pp_32k
        # pp_values_32k[k] = [list(selector_32k.get_support(indices=True)), pp_32k[selector_32k.get_support(indices=True)]]
        # pp_values_1k[k] = [list(selector_1k.get_support(indices=True)), pp_1k[selector_1k.get_support(indices=True)]]
        pp_values_1k[k] = pp_1k
        if k_val == 5:
            k5_1k_feats = selector_1k.get_support(indices=True)
            k5_32k_feats = selector_32k.get_support(indices=True)
    # print(k5_1k_feats)
    # print(k5_32k_feats)
    # print(k5_32k_feats.intersection(k5_1k_feats))
    intersect_set = set(k5_32k_feats).intersection(set(k5_1k_feats))


    X_reduced_test[0] = x_test[:, k5_32k_feats]
    X_reduced_test[1] = x_test[:, k5_1k_feats]
    # do the same for the 1K dataset
    models_1k[0] = classfier_choices[i]() # training the (k = 5, 1k) model
    models_1k[0].fit(X_reduced_1k[0], y_1k) # training the (k = 5, 1k) model on the (k = 5, 1k) dataset
    models_32k[0] = classfier_choices[i]() # training the (k = 5, 32k) model
    models_32k[0].fit(X_reduced_32k[0], y_train) # training the (k = 5, 32k) model with the (k = 5, 32k) dataset
    y_temp = models_32k[0].predict(X_reduced_test[0])
    accuracies_32k[0] = accuracy(confusion_matrix(y_test, y_temp, labels=[0, 1, 2, 3]))
    y_temp = models_1k[0].predict(X_reduced_test[1])
    accuracies_1k[0] = accuracy(confusion_matrix(y_test, y_temp, labels=[0, 1, 2, 3]))






    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.
        # for each number of features k_feat, write the p-values for
        # that number of features:
        for k in range(0, len(pp_values_32k)):
            outf.write(f'{k_vals[k]} p-values: {[round(p_val, 4) for p_val in pp_values_32k[k]]}\n')

        outf.write(f'Accuracy for 1k: {accuracies_1k[0]:.4f}\n')
        outf.write(f'Accuracy for full dataset: {accuracies_32k[0]:.4f}\n')
        outf.write(f'Chosen feature intersection: {list(intersect_set)}\n')
        outf.write(f'Top-5 at higher: {list(k5_32k_feats)}\n')

        pass


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
        '''

    classfier_choices = [SGDClassifier, GaussianNB, RandomForestClassifier, MLPClassifier, AdaBoostClassifier]
    X_all = np.concatenate((X_test, X_train))
    y_all = np.concatenate((y_test, y_train))
    kf = KFold(n_splits=5, shuffle=True)
    kf.get_n_splits(X_all)
    k_fold_accuracy_table = [[], [], [], [], []]
    k_fold_accuracy_table_transpose = [[], [], [], [], []]
    j = 0
    for train_index, test_index in kf.split(X_all):
        for k in range(0, 5):
            classifier = classfier_choices[k]()
            if k == 2:
                classifier = classfier_choices[k](max_depth=5, n_estimators=10)
            elif k == 3:
                classifier = classfier_choices[k](alpha=0.5)
            classifier.fit(X_all[train_index, :], y_all[train_index])
            y_pred = classifier.predict(X_all[test_index, :])
            acc = accuracy(confusion_matrix(y_all[test_index], y_pred, labels=[0, 1, 2, 3]))
            k_fold_accuracy_table[j].append(acc)
            k_fold_accuracy_table_transpose[k].append(acc)
        j = j + 1
    non_best_index_set = set()
    p_values = []
    for k in range (0, 4):
        if k != i:
            non_best_index_set.add(k)
    for index in sorted(list(non_best_index_set)):
        p_values.append(ttest_rel(k_fold_accuracy_table_transpose[index], k_fold_accuracy_table_transpose[i]).pvalue)

    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        # for each fold:
        for fold in k_fold_accuracy_table:
            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in fold]}\n')
        outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()

    # TODO: load data and split into train and test.
    all_data = None
    with np.load('feats.npz') as data:
        all_data = data["arr_0"]
    y_all = all_data[:,173]
    x_all = all_data[:,:173]
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=0)

    # test code  ==========================================
    iBest = 4
    X_discarded, X_1k, y_discarded, y_1k = train_test_split(x_train, y_train, test_size=1 / 32, random_state=0)
    # test code end ==========================================
    np.random.seed(401)
    # iBest = class31(args.output_dir, x_train, x_test, y_train, y_test)
    np.random.seed(401)
    # X_1k, y_1k = class32(args.output_dir, x_train, x_test, y_train, y_test, iBest=iBest)
    np.random.seed(401)
    # class33(args.output_dir, x_train, x_test, y_train, y_test, iBest, X_1k, y_1k)
    np.random.seed(401)
    class34(args.output_dir, x_train, x_test, y_train, y_test, i=iBest)

    # TODO : complete each classification experiment, in sequence.