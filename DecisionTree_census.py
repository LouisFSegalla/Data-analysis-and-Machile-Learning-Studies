# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 20:24:42 2022

@author: luisf
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree 
import matplotlib.pyplot as plt
import pickle #used to save and load the data
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix 

with open(r'saved_databases/census.pkl','rb') as f:
    x_census_training, y_census_training, x_census_test, y_census_test = pickle.load(f)


#creating the decision tree object
censusTree = DecisionTreeClassifier(criterion='entropy',random_state=0)
#training the tree with the dataset
censusTree.fit(x_census_training,y_census_training)

#Using our training dataset with the trained tree
previsions = censusTree.predict(x_census_test)

#Accuracy test
accuracy = accuracy_score(y_census_test,previsions)

#Creating a confusion matrix of the data
cm = ConfusionMatrix(censusTree)
cm.fit(x_census_training,y_census_training)
cm.score(x_census_test,y_census_test)

#Some more info about the trained tree
print(classification_report(y_census_test,previsions))

