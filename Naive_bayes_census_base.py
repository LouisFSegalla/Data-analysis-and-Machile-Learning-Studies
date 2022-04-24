# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 14:15:14 2022

@author: luisf
"""

#importing the necessary libraries
import pickle #used to save the data
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix 

#upload the data that was previously treated
with open(r'saved_databases/census.pkl','rb') as f:
    x_census_training, y_census_training, x_census_test, y_census_test = pickle.load(f)
    
#Train our model
naive_census = GaussianNB()
naive_census.fit(x_census_training,y_census_training)

#Test the resultant model
previsions = naive_census.predict(x_census_test)

accuracy = accuracy_score(y_census_test,previsions)
confusion_matrix_output = confusion_matrix(y_census_test,previsions)
print(classification_report(y_census_test,previsions))

#Using yelowbrick library to visualize the confusion matrix
cm = ConfusionMatrix(naive_census)
cm.fit(x_census_training,y_census_training)
cm.score(x_census_test,y_census_test)
