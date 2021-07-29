#IMPORTAZIONI VARIE
import sys
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
import numpy as np
import seaborn as sns
from graphviz import Source
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import os
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

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
#last.replace({1: 'rotto', 0: 'non rotto'}, inplace=True)


#CORRELAZIONI TRA FEATURE
pd.plotting.scatter_matrix(dataset.drop(['L', 'M', 'H', 'Machine failure'], axis=1, inplace=False), figsize=(14,22))
plt.show()
#sembra esserci una correlazione tra air temperature e process temperature e infatti è così sicché process temperature è
#calcolata a partire da air temperature e una correlazione inversa tra rotational speed e torque (infatti sono inversamente proporzionali)

#print(np.corrcoef(dataset['Air temperature [K]'], dataset['Process temperature [K]'])) #-->0.87
#print(np.corrcoef(dataset['Rotational speed [rpm]'], dataset['Torque [Nm]'])) #-->-0.87

plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(dataset.drop(['L', 'M', 'H', 'Machine failure'], axis=1, inplace=False).corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap')
plt.show()


#visualizzazione per cercare eventuali outliers
#Avendo i record più di 3 dimensioni, tutte importanti per individuarli, non possono essere stampati
#Analizzo le eventuali anomalie nelle singole feature

dataset.drop(['L', 'M', 'H', 'Machine failure'], axis=1, inplace=False).plot(subplots=True, figsize=(16,6))
plt.show()
#--> non ci sono anomalie nei valori delle singole feature


#############################################

#INSERIRE QUI CHIAMATE A FUNZIONI CHE SI OCCUPANO DELL'ADDESTRAMENTO E DELLA VALUTAZIONI DELLE PRESTAZIONI DI TUTTI I MODELLI. (QUESTIONE DI ORDINE (MENTALE ANCHE))


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

def decision_tree(dataset):

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

    dataset.drop(['L', 'M', 'H'], axis = 1, inplace = True) # non venivano mai considerati nell'albero (ce ne siamo accorti dopo alcune prove)
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]

# split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.65, random_state = 42)

# albero di decisione 

# euristica per i pesi dato lo sbilanciamento del dataset

    nonfailure_count, failure_count = dataset.groupby('Machine failure').size()
    ratio = nonfailure_count / failure_count
    weights = {0: 1.0, 1: ratio}

# search with grid_search_cv

    params = {'min_samples_leaf': [30, 40, 50]}
    cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 42)
    grid_search_cv = GridSearchCV(DecisionTreeClassifier(max_depth = 4, class_weight = weights, criterion = 'entropy', random_state=42), params, verbose=1, cv = cv, scoring='f1_weighted')
    grid_search_cv.fit(X_train, Y_train)
    draw_tree("tree_pm_grid_search.dot", grid_search_cv.best_estimator_, dataset.columns.values[:-1], ['Non rotto', 'Rotto'])

# test prestazioni

    Y_pred = grid_search_cv.predict(X_test)
    print('Accuracy score of decisione tree: ', accuracy_score(Y_test, Y_pred))
    print('Best params after grid search applied to decision tree: ', grid_search_cv.best_params_)


    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    print("TrueNegative: %d" % tn)
    print("FalsePostive: %d" % fp)
    print("FalseNegative: %d" % fn)
    print("TruePositive: %d" % tp)


# I nostri risultati per l'albero di decisione sono:
# Il dataset è fortemente sbilanciato a favore delle istanze negative (il 3.39 % delle istanze sono positive).
# Per questo motivo, abbiamo deciso di pesare i dati di input (utilizzando come euristica per i pesi il rapporto tra il 
# numero di istanze negative e il numero di istanze positive). Inoltre abbiamo utilizzato una score F2 per dare maggiore importanza ai falsi negativi, problema grave in ambito industriale
# e di manutenzione predittiva
# Dopodiché abbiamo usato RepeteadStratifiedKFold passato come parametro a una GridSearchCV per la crossvalidazione a caccia della migliore configurazione possibile di iperparametri
# Questo ci ha permesso di determinare in maniera automatica alcuni degli iperparametri.
# L'accuracy score è circa di 0.89.
# fn e tp erano circa 5 e 100, quindi un risultato buono considerando quanto è sbilanciato il dataset.

# Support vector machines

from sklearn.svm import SVC
from sklearn.svm import LinearSVC

def support_vector_machines(dataset):
    # prepara test e training data
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.65, random_state = 42)

    tuned_parameters = [{'C': [0.01, 0.1, 1, 10, 100],
                         'gamma': [0.5, 1,2,3,4]}]
    cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 42)
    clf = GridSearchCV(SVC(kernel='rbf'), tuned_parameters, cv=10, scoring='accuracy')
    clf.fit(X_train, Y_train)
    print(clf.best_params_)

support_vector_machines(dataset)
