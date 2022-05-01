# -*- coding: utf-8 -*-
"""
Created on Sun May  1 08:08:16 2022

@author: luisf
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree 
import matplotlib.pyplot as plt
import pickle #used to save and load the data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix 
###############################################################################

with open(r'saved_databases/census.pkl','rb') as f:
    x_census_training, y_census_training, x_census_test, y_census_test = pickle.load(f)
    
    
randForestCensus = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)
randForestCensus.fit(x_census_training,y_census_training)

predictions = randForestCensus.predict(x_census_test)

accuracy = accuracy_score(y_census_test,predictions)
confusion_matrix_output = confusion_matrix(y_census_test,predictions)
print(classification_report(y_census_test, predictions))

#Creating a confusion matrix of the data
cm = ConfusionMatrix(randForestCensus)
cm.fit(x_census_training,y_census_training)
cm.score(x_census_test, y_census_test)
