# -*- coding: utf-8 -*-
"""
Created on Sun May  8 11:01:16 2022

@author: luisf
"""

from sklearn.neighbors import KNeighborsClassifier
import pickle #used to save and load the data
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix 

with open(r'saved_databases/census.pkl','rb') as f:
    x_census_training, y_census_training, x_census_test, y_census_test = pickle.load(f)
    

KNN_census = KNeighborsClassifier(n_neighbors=10)
KNN_census.fit(x_census_training,y_census_training)

previsions = KNN_census.predict(x_census_test)

accuracy = accuracy_score(y_census_test,previsions)

print(classification_report(y_census_test,previsions))

#Creating a confusion matrix of the data
cm = ConfusionMatrix(KNN_census)
cm.fit(x_census_training,y_census_training)
cm.score(x_census_test,y_census_test)