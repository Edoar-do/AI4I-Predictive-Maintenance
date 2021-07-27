import pandas as pd
import matplotlib as mpl
import numpy as np
import sklearn as skl
import numpy as np

#lettura dataset
dataset = pd.read_csv('ai4i2020.csv')

#DATA ANALYSIS:
#riduzione dimensionalità
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
dataset = dataset.join(one_hot)
#dataset.replace({'L': 1 << 0, 'M': 1 << 1, 'H': 1 << 2}, inplace=True)
print(dataset.columns)
#eventuali correlazioni tra feature
pd.plotting.scatter_matrix(dataset.drop(['L', 'M', 'H', 'Machine failure'], axis=1, inplace=False))
mpl.pyplot.show()
#sembra esserci una correlazione tra air temperature e process temperature e infatti è così sicché process temperature è
#calcolata a partire da air temperature e una correlazione inversa tra rotational speed e torque (infatti sono inversamente proporzionali)
print(np.corrcoef(dataset['Air temperature [K]'], dataset['Process temperature [K]'])) #-->0.87
print(np.corrcoef(dataset['Rotational speed [rpm]'], dataset['Torque [Nm]'])) #-->-0.87


#visualizzazione per cercare eventuali outliers
#Avendo i record più di 3 dimensioni, tutte importanti per individuarli, non possono essere stampati
#Analizzo le eventuali anomalie nelle singole feature
dataset.drop(['L', 'M', 'H', 'Machine failure'], axis=1, inplace=False).plot(subplots=True)
mpl.pyplot.show()
#--> non ci sono anomalie nei valori delle singole feature



#ADDESTRAMENTO DI ALCUNI MODELLI
#1. Albero di decisione
#2. Rete neurale
#3. SVM
#4. RandomForest
#5. Bayes naive (non c'è la condizionale indipendenza tra attributi -process temperature dipende da air temperature- ma i risultati potrebbero comunque essere buoni.
#nel caso si prova a togliere una delle due e vedere i risultati)
#6. eventuali altri modelli...





