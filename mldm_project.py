import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from graphviz import Source
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import os
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
#from autosklearn.classification import AutoSklearnClassifier
import pickle


enable_plot_tree = False
enable_plot_confusion_matrix = True

#os.environ["PATH"] += os.pathsep + 'C:/Utenti/jed/Anaconda3/envs/keras/Library/bin/graphviz/'
dataset = pd.read_csv('ai4i2020.csv')

#DATA ANALYSIS:
## dimensionality reduction
dataset.drop("Product ID", axis=1, inplace=True)
dataset.drop("UDI", axis=1, inplace=True)
dataset.drop("TWF", axis=1, inplace=True)
dataset.drop("HDF", axis=1, inplace=True)
dataset.drop("PWF", axis=1, inplace=True)
dataset.drop("OSF", axis=1, inplace=True)
dataset.drop("RNF", axis=1, inplace=True)

# Checking for NaN values in the dataset can be skipped, as certified by dataset owners

# Since data comes from industrial data collection, is it possible to have non-informative duplicates
pd.DataFrame.drop_duplicates(dataset, subset=dataset.columns.difference(['Machine failure']), inplace=True)

#one hot encoding of categorical feature
one_hot = pd.get_dummies(dataset['Type'])
dataset = dataset.drop('Type', axis=1)
last = dataset.iloc[:, -1]
dataset = dataset.drop('Machine failure', axis = 1)
dataset = dataset.join(one_hot)
dataset = dataset.join(last)

#searching for strong correlation in order to reduce dimensionality once again

#pd.plotting.scatter_matrix(dataset.drop(['L', 'M', 'H', 'Machine failure'], axis=1, inplace=False), figsize=(14,22))
#plt.show()

#strong positibe correlation between air temperature e process temperature 
#strong negative correlation between rotational speed e torque

#plt.figure(figsize=(16, 6))
#heatmap = sns.heatmap(dataset.drop(['L', 'M', 'H', 'Machine failure'], axis=1, inplace=False).corr(), vmin=-1, vmax=1, annot=True)
#heatmap.set_title('Correlation Heatmap')
#plt.show()

#############################################

# TRAINING SOME MODELS

#1. decision tree
#2. MLP
#3. SVM
#4. RandomForest
#5. Bayes naive (there is no conditional independence between features but results might be good. If not, we can remove one of two correlated features and try again
#6. possible other models


PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def draw_tree(filename, tree, dataset):
    feature_names = dataset.columns.values[:-1]
    class_names = ['Non rotto', 'Rotto']
    export_graphviz(
        tree,
        out_file=os.path.join(IMAGES_PATH, filename),
        feature_names=feature_names,
        class_names=class_names,
        rounded=True,
        filled=True
    )

    Source.from_file(os.path.join(IMAGES_PATH, filename)).render(view=True)

np.random.seed(42)

def train_test_split_standard_scaler(X, Y, train_size, random_state):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.65, random_state=42)
    scaler = StandardScaler() 
    scaler.fit(X_train)  
    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test)
    return (X_train, X_test, Y_train, Y_test)

def decision_treeWeighted(dataset):
    # I build the dataset into the inputs and targets that will be used for training.

    dataset = dataset.drop(['L', 'M', 'H'], axis = 1) # were never considered in the tree (we realized this after some trials)
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]

    # split data
    X_train, X_test, Y_train, Y_test = train_test_split_standard_scaler(X, Y, train_size = 0.65, random_state = 42)

    # decision tree
    # heuristics for weights given the unbalanced dataset

    #nonfailure_count, failure_count = dataset.groupby('Machine failure').size()
    #ratio = nonfailure_count / failure_count
    #weights = {0: 1.0, 1: ratio}

    tree = DecisionTreeClassifier(max_depth = 4, class_weight = 'balanced', criterion = 'entropy', random_state=42, min_samples_leaf=30)
    dizionario = cross_validate(tree, X_train, Y_train, cv=5, n_jobs=-1, scoring='recall', return_estimator=True)

    Y_pred = dizionario['estimator'][0].predict(X_test)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    print("TrueNegative: %d" % tn)
    print("FalsePostive: %d" % fp)
    print("FalseNegative: %d" % fn)
    print("TruePositive: %d" % tp)
    if enable_plot_tree:
        draw_tree("tree-weighted", dizionario['estimator'][0], dataset)

    if enable_plot_confusion_matrix:
        plot_confusion_matrix(dizionario['estimator'][0], X_test, Y_test)
        plt.title('Confusion Matrix Weighted Tree')
        plt.show()

    #Results obtained with model cross validation (WEIGHTED) and performance calculation with confusion matrix on previously separated testing data:
    #Recall about 5.7% (on '1' values) -> not an exceptional value given the industrial context
    #Specificity about 9.34

def svm_weighted(dataset):
    dataset = dataset.drop(['L', 'M', 'H'], axis=1)  # were never considered in the tree (we realized this after some trials)
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]

    # split data
    X_train, X_test, Y_train, Y_test = train_test_split_standard_scaler(X, Y, train_size=0.65, random_state=42)

    # heuristics for weights given the unbalanced dataset.

    svm = SVC(kernel='rbf', C=float('inf'), gamma=1, class_weight='balanced', random_state=42)
    cv = 5
    dizionario = cross_validate(svm, X_train, Y_train, cv=cv, n_jobs=-1, scoring='recall', return_estimator=True)

    Y_pred = dizionario['estimator'][0].predict(X_test)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    print("TrueNegative: %d" % tn)
    print("FalsePostive: %d" % fp)
    print("FalseNegative: %d" % fn)
    print("TruePositive: %d" % tp)

    if enable_plot_confusion_matrix:
        plot_confusion_matrix(dizionario['estimator'][0], X_test, Y_test)
        plt.title('Confusion Matrix Weighted SVM')
        plt.show()


    #recall=0 e specificity=3394/3395 -> risultati non accettabili

def svm_sampling(dataset):
    dataset = dataset.drop(['L', 'M', 'H'], axis=1)
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split_standard_scaler(X, Y, train_size=0.65, random_state=42)

    svm = SVC(kernel='rbf', C=float('inf'), gamma=1, class_weight='balanced', random_state=42)
    over = SMOTE(sampling_strategy=0.45, random_state=42)
    under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    X_train, Y_train = over.fit_resample(X_train, Y_train)
    X_train, Y_train = under.fit_resample(X_train, Y_train)
    cv = 5
    dizionario = cross_validate(svm, X_train, Y_train, cv=cv, n_jobs=-1, scoring='recall', return_estimator=True)

    Y_pred = dizionario['estimator'][0].predict(X_test)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    print("TrueNegative: %d" % tn)
    print("FalsePostive: %d" % fp)
    print("FalseNegative: %d" % fn)
    print("TruePositive: %d" % tp)
    if enable_plot_confusion_matrix:
        plot_confusion_matrix(dizionario['estimator'][0], X_test, Y_test)
        plt.title('Confusion Matrix SVM with Sampling')
        plt.show()

    # recall = 2/105 and specificity = 3391/3395 -> terribly unacceptable results

def decisionTree_sampling(dataset):
    dataset = dataset.drop(['L', 'M', 'H'], axis=1)
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split_standard_scaler(X, Y, train_size=0.65, random_state=42)


    tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=30, class_weight='balanced')
    over = SMOTE(sampling_strategy=0.45, random_state=42)
    under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    X_train, Y_train = over.fit_resample(X_train, Y_train)
    X_train, Y_train = under.fit_resample(X_train, Y_train)
    cv = 5
    dizionario = cross_validate(tree, X_train, Y_train, cv=cv, n_jobs=-1, scoring='recall', return_estimator=True)

    Y_pred = dizionario['estimator'][0].predict(X_test)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    print("TrueNegative: %d" % tn)
    print("FalsePostive: %d" % fp)
    print("FalseNegative: %d" % fn)
    print("TruePositive: %d" % tp)
    if enable_plot_confusion_matrix:
        plot_confusion_matrix(dizionario['estimator'][0], X_test, Y_test)
        plt.title('Confusion Matrix Decision Tree with sampling')
        plt.show()

    if enable_plot_tree:
        draw_tree("tree-sampling", dizionario['estimator'][0], dataset)
    # Results obtained by cross validation of model trained on over and undersampling and performance calculation with confusion matrix on previously separated testing data:
    # Recall about 8.6% (on '1' values) -> not an exceptional value given the industrial context.
    # Specificity about 9.2%
    #Since what matters is recall and not specificity in this context, the decision tree with weights is better than the one with over-under sampling

def randomForest_weighted(dataset):
    dataset = dataset.drop(['L', 'M', 'H'], axis=1)
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]

    # split data
    X_train, X_test, Y_train, Y_test = train_test_split_standard_scaler(X, Y, train_size=0.65, random_state=42)

    randomForest = RandomForestClassifier(n_jobs=-1, criterion='entropy', n_estimators=1000, class_weight='balanced_subsample',
                                         random_state=42, max_samples=0.5, max_leaf_nodes=16)
    cv = 5
    dizionario = cross_validate(randomForest, X_train, Y_train, cv=cv, n_jobs=-1, scoring='recall', return_estimator=True)

    Y_pred = dizionario['estimator'][0].predict(X_test)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    print("TrueNegative: %d" % tn)
    print("FalsePostive: %d" % fp)
    print("FalseNegative: %d" % fn)
    print("TruePositive: %d" % tp)
    if enable_plot_confusion_matrix:
        plot_confusion_matrix(dizionario['estimator'][0], X_test, Y_test)
        plt.title('Confusion Matrix Weighted Random Forest')
        plt.show()

    #recall 10/105, specificity=3118/3118+277

def samples_randomForest(dataset):
    dataset = dataset.drop(['L', 'M', 'H'],axis=1)
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]

    # split data
    X_train, X_test, Y_train, Y_test = train_test_split_standard_scaler(X, Y, train_size=0.65, random_state=42)

    randomForest = RandomForestClassifier(n_jobs=-1, criterion='entropy', n_estimators=1000,random_state=42,
                                          max_samples=0.5, max_leaf_nodes=16, class_weight='balanced_subsample')
    over = SMOTE(sampling_strategy=0.45, random_state=42)
    under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    X_train, Y_train = over.fit_resample(X_train, Y_train)
    X_train, Y_train = under.fit_resample(X_train, Y_train)

    cv = 5
    dizionario = cross_validate(randomForest, X_train, Y_train, cv=cv, n_jobs=-1, scoring='recall', return_estimator=True)

    Y_pred = dizionario['estimator'][0].predict(X_test)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    print("TrueNegative: %d" % tn)
    print("FalsePostive: %d" % fp)
    print("FalseNegative: %d" % fn)
    print("TruePositive: %d" % tp)

    if enable_plot_confusion_matrix:
        plot_confusion_matrix(dizionario['estimator'][0], X_test, Y_test)
        plt.title('Confusion Matrix Random Forest with sampling')
        plt.show()
    print()
    print('FEATURE IMPORTANCE')

    for name, score in zip(dataset.columns, dizionario['estimator'][0].feature_importances_):
        print(name, score)



    #recall = 5/105, specificity=3039/3039+356 MODELLO MIGLIORE MAI VISTO!!!!
    #The “balanced_subsample” mode is the same as “balanced” except that weights are computed based on the bootstrap sample for every tree grown.

def decisionTreeWithMostImportantFeature(dataset):
    #I build the dataset into the inputs and targets that will be used for training

    dataset = dataset.drop(['L', 'M', 'H', 'Air temperature [K]', 'Process temperature [K]'],axis=1)  # they were never considered in the tree (we realized this after a few trials)
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]

    # split data
    X_train, X_test, Y_train, Y_test = train_test_split_standard_scaler(X, Y, train_size=0.65, random_state=42)

    # tree (d) decision
    # heuristics for weights given the unbalanced dataset


    # nonfailure_count, failure_count = dataset.groupby('Machine failure').size()
    # ratio = nonfailure_count / failure_count
    # weights ="{0: 1.0, 1: ratio}

    tree = DecisionTreeClassifier(max_depth=4, class_weight='balanced', criterion='entropy', random_state=42,
                                  min_samples_leaf=30)
    cv = 5
    dizionario = cross_validate(tree, X_train, Y_train, cv=cv, n_jobs=-1, scoring='recall', return_estimator=True)

    Y_pred = dizionario['estimator'][0].predict(X_test)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    print("TrueNegative: %d" % tn)
    print("FalsePostive: %d" % fp)
    print("FalseNegative: %d" % fn)
    print("TruePositive: %d" % tp)
    if enable_plot_confusion_matrix:
        plot_confusion_matrix(dizionario['estimator'][0], X_test, Y_test)
        plt.title('Confusion Matrix Decision Tree with most important features')
        plt.show()

    if enable_plot_tree:
        draw_tree("most-important", dizionario['estimator'][0], dataset)

    # recall=3.81%, specificity=17.67%
    # thanks to random forest with samples we found the most important features and now we used them to train a very simple model like a tree
    # to see if there were any performance improvements, and there were.

def gaussian_naive(dataset):
    dataset = dataset.drop(['L', 'M', 'H'], axis=1)  # were never considered in the tree (we realized this after some trials)
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]

    # split data
    X_train, X_test, Y_train, Y_test = train_test_split_standard_scaler(X, Y, train_size=0.65, random_state=42)

    naive = GaussianNB()
    cv = 5
    dizionario = cross_validate(naive, X_train, Y_train, cv=cv, n_jobs=-1, scoring='recall', return_estimator=True)

    Y_pred = dizionario['estimator'][0].predict(X_test)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    print("TrueNegative: %d" % tn)
    print("FalsePostive: %d" % fp)
    print("FalseNegative: %d" % fn)
    print("TruePositive: %d" % tp)
    if enable_plot_confusion_matrix:
        plot_confusion_matrix(dizionario['estimator'][0], X_test, Y_test)
        plt.title('Confusion Matrix Gaussian Naive Bayes')
        plt.show()

def gaussian_naive_sampling(dataset):
    dataset = dataset.drop(['L', 'M', 'H'], axis=1)
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split_standard_scaler(X, Y, train_size=0.65, random_state=42)

    naive = GaussianNB()
    over = SMOTE(sampling_strategy=0.45, random_state=42)
    under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    X_train, Y_train = over.fit_resample(X_train, Y_train)
    X_train, Y_train = under.fit_resample(X_train, Y_train)
    cv = 5
    dizionario = cross_validate(naive, X_train, Y_train, cv=cv, n_jobs=-1, scoring='recall', return_estimator=True)

    Y_pred = dizionario['estimator'][0].predict(X_test)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    print("TrueNegative: %d" % tn)
    print("FalsePostive: %d" % fp)
    print("FalseNegative: %d" % fn)
    print("TruePositive: %d" % tp)
    if enable_plot_confusion_matrix:
        plot_confusion_matrix(dizionario['estimator'][0], X_test, Y_test)
        plt.title('Confusion Matrix Gaussian Naive Bayes with sampling')
        plt.show()

def reteNeurale(dataset):
    dataset = dataset.drop(['L', 'M', 'H'], axis=1)
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split_standard_scaler(X, Y, train_size=0.75, random_state=42)

    over = SMOTE(sampling_strategy=0.45, random_state=42)
    under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    X_train, Y_train = over.fit_resample(X_train, Y_train)
    X_train, Y_train = under.fit_resample(X_train, Y_train)

    ann = MLPClassifier(solver='adam', validation_fraction=0.25, early_stopping=True, activation='relu', learning_rate='adaptive', n_iter_no_change=10,  random_state=42, alpha=1e-6, max_iter=500)
    cv = 5
    dizionario = cross_validate(ann, X_train, Y_train, cv=cv, n_jobs=-1, scoring='recall', return_estimator=True)

    Y_pred = dizionario['estimator'][0].predict(X_test)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    print("TrueNegative: %d" % tn)
    print("FalsePostive: %d" % fp)
    print("FalseNegative: %d" % fn)
    print("TruePositive: %d" % tp)
    if enable_plot_confusion_matrix:
        plot_confusion_matrix(dizionario['estimator'][0], X_test, Y_test)
        plt.title('Confusion Matrix ANN')
        plt.show()

# def auto_sklearn(dataset):
#     dataset = dataset.drop(['L', 'M', 'H'], axis=1)
#     X = dataset.iloc[:, :-1]
#     Y = dataset.iloc[:, -1]
#     X_train, X_test, Y_train, Y_test = train_test_split_standard_scaler(X, Y, train_size=0.65, random_state=42)
#
#     # n_jobs e' buggato
#     autosk = AutoSklearnClassifier(
#             include_preprocessors=["no_preprocessing", ],
#             exclude_preprocessors=None,
#             memory_limit=3000,
#     )
#     over = SMOTE(sampling_strategy=0.45, random_state=42)
#     under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
#     X_train, Y_train = over.fit_resample(X_train, Y_train)
#     X_train, Y_train = under.fit_resample(X_train, Y_train)
#     fname = "auto_sklearn"
#     #autosk.fit(X_train, Y_train)
#     #with open(fname, 'wb') as file:
#         #pickle.dump(autosk, file)
#     with open(fname, "rb") as file:
#         autosk = pickle.load(file)
#     Y_pred = autosk.predict(X_test)
#     tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
#     print("TrueNegative: %d" % tn)
#     print("FalsePostive: %d" % fp)
#     print("FalseNegative: %d" % fn)
#     print("TruePositive: %d" % tp)
#     if enable_plot_confusion_matrix:
#         plot_confusion_matrix(autosk, X_test, Y_test)
#         plt.title('Confusion Matrix AutoSklearn Model')
#         plt.show()

# print('DECISION TREE PESATO')
# decision_treeWeighted(dataset)
# print()
# print('DECISION TREE WITH SAMPLING')
# decisionTree_sampling(dataset)
# print()
# print('SVM PESATO')
# svm_weighted(dataset)
# print()
# print('SVM CON SAMPLING')
# svm_sampling(dataset)
# print()
# print("NAIVE BAYES GAUSSIAN CLASSIFIER")
# gaussian_naive(dataset)
# print()
# print("NAIVE BAYES GAUSSIAN CLASSIFIER WITH SAMPLING")
# gaussian_naive_sampling(dataset)
# print()
# print('ARITIFICIAL NEURAL NETWORK')
# reteNeurale(dataset)
# print()
# print('RANDOM FOREST PESATA')
# randomForest_weighted(dataset)
# print()
print('RANDOM FOREST WITH SAMPLING')
samples_randomForest(dataset)
print()
print('MOST IMPORTANT FEATURE WEIGHTED DECISION TREE')
decisionTreeWithMostImportantFeature(dataset)
print()



# print('AUTO SKLEARN')
# auto_sklearn(dataset)
# print()
