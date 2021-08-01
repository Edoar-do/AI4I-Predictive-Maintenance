#IMPORTAZIONI VARIE
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from graphviz import Source
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import os
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

#os.environ["PATH"] += os.pathsep + 'C:/Utenti/jed/Anaconda3/envs/keras/Library/bin/graphviz/'
#LETTURA DATASET
dataset = pd.read_csv('ai4i2020.csv')

#DATA ANALYSIS:
#riduzione dimensionalità: tolgo feature inutili in partenza per gli scopi del progetto
dataset.drop("Product ID", axis=1, inplace=True)
dataset.drop("UDI", axis=1, inplace=True)
dataset.drop("TWF", axis=1, inplace=True)
dataset.drop("HDF", axis=1, inplace=True)
dataset.drop("PWF", axis=1, inplace=True)
dataset.drop("OSF", axis=1, inplace=True)
dataset.drop("RNF", axis=1, inplace=True)

#Non ci sono valori NaN, undefined o null all'interno del dataset. Non occorre fare una ricerca perché la loro assenza
#è certificata dalla descrizione del dataset fornita dall'autore dell'articolo

#Togliere i duplicati nel dataset senza considerare la label Machine Failure (per vedere se ci sono record discordi)
pd.DataFrame.drop_duplicates(dataset, subset=dataset.columns.difference(['Machine failure']), inplace=True)
#--> non ci sono proprio record duplicati

#one hot encoding dei valori L, M, H dell'attributo Type
one_hot = pd.get_dummies(dataset['Type'])
dataset = dataset.drop('Type', axis=1)
last = dataset.iloc[:, -1]
dataset = dataset.drop('Machine failure', axis = 1)
dataset = dataset.join(one_hot)
dataset = dataset.join(last)

#CORRELAZIONI TRA FEATURE

#pd.plotting.scatter_matrix(dataset.drop(['L', 'M', 'H', 'Machine failure'], axis=1, inplace=False), figsize=(14,22))
#plt.show()

#sembra esserci una correlazione tra air temperature e process temperature e infatti è così sicché process temperature è
#calcolata a partire da air temperature e una correlazione inversa tra rotational speed e torque (infatti sono inversamente proporzionali)

#plt.figure(figsize=(16, 6))
#heatmap = sns.heatmap(dataset.drop(['L', 'M', 'H', 'Machine failure'], axis=1, inplace=False).corr(), vmin=-1, vmax=1, annot=True)
#heatmap.set_title('Correlation Heatmap')
#plt.show()

#visualizzazione per cercare eventuali outliers
#Avendo i record più di 3 dimensioni, tutte importanti per individuarli, non possono essere stampati
#Analizzo le eventuali anomalie nelle singole feature

#dataset.drop(['L', 'M', 'H', 'Machine failure'], axis=1, inplace=False).plot(subplots=True, figsize=(16,6))
#plt.show()
#--> non ci sono anomalie nei valori delle singole feature


#############################################

# ADDESTRAMENTO DI ALCUNI MODELLI

#1. Albero di decisione
#2. Rete neurale
#3. SVM
#4. RandomForest
#5. Bayes naive (non c'è la condizionale indipendenza tra attributi -process temperature dipende da air temperature- ma i risultati potrebbero comunque essere buoni.
#nel caso si prova a togliere una delle due e vedere i risultati)
#6. eventuali altri modelli...


PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

np.random.seed(42)

def decision_treeWeighted(dataset):

    # draw tree

    def draw_tree(filename, tree, feature_names, class_names):
        export_graphviz(
                tree,
                out_file=os.path.join(IMAGES_PATH, filename),
                feature_names=feature_names,
                class_names=class_names,
                rounded=True,
                filled=True
                )

        Source.from_file(os.path.join(IMAGES_PATH, filename)).render(view = True)

    # Costruisco il dataset negli input e nei target che serviranno per l'addestramento

    dataset = dataset.drop(['L', 'M', 'H'], axis = 1) # non venivano mai considerati nell'albero (ce ne siamo accorti dopo alcune prove)
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]

    # split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.65, random_state = 42)

    # albero di decisione
    # euristica per i pesi dato lo sbilanciamento del dataset

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

    #Risultati ottenuti con cross validazione del modello (PESATO) e calcolo delle prestazioni con matrice di confusione su dati di testing precedentemente separati:
    #Recall circa del 5.7% (sui valori '1') -> valore non eccezionale dato il contesto industriale
    #Specificity circa del 9.34

def svm_weighted(dataset):
    dataset = dataset.drop(['L', 'M', 'H'], axis=1)  # non venivano mai considerati nell'albero (ce ne siamo accorti dopo alcune prove)
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]

    # split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.65, random_state=42)

    # euristica per i pesi dato lo sbilanciamento del dataset

    svm = SVC(kernel='rbf', C=float('inf'), gamma=1, class_weight='balanced', random_state=42)
    dizionario = cross_validate(svm, X_train, Y_train, cv=5, n_jobs=-1, scoring='recall', return_estimator=True)

    Y_pred = dizionario['estimator'][0].predict(X_test)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    print("TrueNegative: %d" % tn)
    print("FalsePostive: %d" % fp)
    print("FalseNegative: %d" % fn)
    print("TruePositive: %d" % tp)

    #recall=0 e specificity=3394/3395 -> risultati non accettabili

def svm_sampling(dataset):
    dataset = dataset.drop(['L', 'M', 'H'], axis=1)
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.65, random_state=42)

    svm = SVC(kernel='rbf', C=float('inf'), gamma=1, class_weight='balanced', random_state=42)
    over = SMOTE(sampling_strategy=0.45, random_state=42)
    under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    X_train, Y_train = over.fit_resample(X_train, Y_train)
    X_train, Y_train = under.fit_resample(X_train, Y_train)

    dizionario = cross_validate(svm, X_train, Y_train, cv=5, n_jobs=-1, scoring='recall', return_estimator=True)

    Y_pred = dizionario['estimator'][0].predict(X_test)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    print("TrueNegative: %d" % tn)
    print("FalsePostive: %d" % fp)
    print("FalseNegative: %d" % fn)
    print("TruePositive: %d" % tp)

    # recall = 2/105 e specificity = 3391/3395 -> risultati terribilmente non accettabili

def decisionTree_sampling(dataset):
    dataset = dataset.drop(['L', 'M', 'H'], axis=1)
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.65, random_state=42)


    tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=30, class_weight='balanced')
    over = SMOTE(sampling_strategy=0.45, random_state=42)
    under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    steps = [('o', over), ('u', under), ('model', tree)]
    pipeline = Pipeline(steps=steps)

    #scores = cross_val_score(pipeline, X_train, Y_train, cv=5, n_jobs=-1, scoring='roc_auc')
    dizionario = cross_validate(pipeline, X_train, Y_train, cv=5, n_jobs=-1, scoring='recall', return_estimator=True)

    Y_pred = dizionario['estimator'][0].predict(X_test)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    print("TrueNegative: %d" % tn)
    print("FalsePostive: %d" % fp)
    print("FalseNegative: %d" % fn)
    print("TruePositive: %d" % tp)

    # Risultati ottenuti con cross validazione del modello allenato su over e undersampling e calcolo delle prestazioni con matrice di confusione su dati di testing precedentemente separati:
    # Recall circa del 8.6% (sui valori '1') -> valore non eccezionale dato il contesto industriale
    # Specificity circa del 9.2%
    #Siccome quello che conta è la recall e non la specificity in questo contesto, l'albero di decisione coi pesi è migliore di quello con over-under sampling

def randomForest_weighted(dataset):
    dataset = dataset.drop(['L', 'M', 'H'], axis=1)
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]

    # split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.65, random_state=42)

    randomForest = RandomForestClassifier(n_jobs=-1, criterion='entropy', n_estimators=1000, class_weight='balanced_subsample',
                                         random_state=42, max_samples=0.5, max_leaf_nodes=16)
    dizionario = cross_validate(randomForest, X_train, Y_train, cv=5, n_jobs=-1, scoring='recall', return_estimator=True)

    Y_pred = dizionario['estimator'][0].predict(X_test)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    print("TrueNegative: %d" % tn)
    print("FalsePostive: %d" % fp)
    print("FalseNegative: %d" % fn)
    print("TruePositive: %d" % tp)

    #recall 10/105, specificity=3118/3118+277

def samples_randomForest(dataset):
    dataset = dataset.drop(['L', 'M', 'H'], axis=1)
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]

    # split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.65, random_state=42)

    randomForest = RandomForestClassifier(n_jobs=-1, criterion='entropy', n_estimators=1000,random_state=42,
                                          max_samples=0.5, max_leaf_nodes=16, class_weight='balanced_subsample')
    over = SMOTE(sampling_strategy=0.45, random_state=42)
    under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    X_train, Y_train = over.fit_resample(X_train, Y_train)
    X_train, Y_train = under.fit_resample(X_train, Y_train)

    dizionario = cross_validate(randomForest, X_train, Y_train, cv=5, n_jobs=-1, scoring='recall', return_estimator=True)

    Y_pred = dizionario['estimator'][0].predict(X_test)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    print("TrueNegative: %d" % tn)
    print("FalsePostive: %d" % fp)
    print("FalseNegative: %d" % fn)
    print("TruePositive: %d" % tp)

    for name, score in zip(dataset.columns, dizionario['estimator'][0].feature_importances_):
        print(name, score)

    #recall = 5/105, specificity=3039/3039+356 MODELLO MIGLIORE MAI VISTO!!!!
    #The “balanced_subsample” mode is the same as “balanced” except that weights are computed based on the bootstrap sample for every tree grown.

def decisionTreeWithMostImportantFeature(dataset):
    # draw tree

    def draw_tree(filename, tree, feature_names, class_names):
        export_graphviz(
            tree,
            out_file=os.path.join(IMAGES_PATH, filename),
            feature_names=feature_names,
            class_names=class_names,
            rounded=True,
            filled=True
        )

        Source.from_file(os.path.join(IMAGES_PATH, filename)).render(view=True)

    # Costruisco il dataset negli input e nei target che serviranno per l'addestramento

    dataset = dataset.drop(['L', 'M', 'H', 'Air temperature [K]', 'Process temperature [K]'],axis=1)  # non venivano mai considerati nell'albero (ce ne siamo accorti dopo alcune prove)
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]

    # split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.65, random_state=42)

    # albero di decisione
    # euristica per i pesi dato lo sbilanciamento del dataset

    # nonfailure_count, failure_count = dataset.groupby('Machine failure').size()
    # ratio = nonfailure_count / failure_count
    # weights = {0: 1.0, 1: ratio}

    tree = DecisionTreeClassifier(max_depth=4, class_weight='balanced', criterion='entropy', random_state=42,
                                  min_samples_leaf=30)
    dizionario = cross_validate(tree, X_train, Y_train, cv=5, n_jobs=-1, scoring='recall', return_estimator=True)

    Y_pred = dizionario['estimator'][0].predict(X_test)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    print("TrueNegative: %d" % tn)
    print("FalsePostive: %d" % fp)
    print("FalseNegative: %d" % fn)
    print("TruePositive: %d" % tp)

    # recall=3.81%, specificity=17.67%
    # grazie alla random forest con samples abbiamo trovato le feature più importanti e adesso le abbiamo usate per allenare un modello semplicissimo come un albero
    # per vedere se ci sono stati miglioramenti delle prestazioni e così è stato.


# print('SVM PESATO')
# svm_weighted(dataset)
# print()
# print('SVM CON SAMPLING')
# svm_sampling(dataset)
# print()
# print('DECISION TREE PESATO')
# decision_treeWeighted(dataset)
# print()
# print('DECISION TREE WITH SAMPLING')
# decisionTree_sampling(dataset)
# print()
# print('random forest pesata')
# randomForest_weighted(dataset)
# print()
# print('random forest sampling')
# samples_randomForest(dataset)
# print()
# print('most important features weightede decision tree')
# decisionTreeWithMostImportantFeature(dataset)
# print()
