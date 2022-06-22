# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 11:21:35 2022

@author: luisf
"""

from sklearn.neural_network import MLPClassifier
import pickle #used to save and load the data
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix 


with open(r'saved_databases/census.pkl','rb') as f:
    x_census_training, y_census_training, x_census_test, y_census_test = pickle.load(f)
    

neural_network_census = MLPClassifier(activation='relu',max_iter=2000,solver='adam',tol=1e-5,hidden_layer_sizes=(55,55))
neural_network_census.fit(x_census_training,y_census_training)

predictions = neural_network_census.predict(x_census_test)

print('accuracy_score = ', accuracy_score(y_census_test,predictions))

#Creating a confusion matrix of the data
cm = ConfusionMatrix(neural_network_census)
cm.fit(x_census_training,y_census_training)
cm.score(x_census_test, y_census_test)

print(classification_report(y_census_test,predictions))