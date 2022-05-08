# -*- coding: utf-8 -*-
"""
Created on Sun May  8 11:01:16 2022

@author: luisf
"""

from sklearn.neighbors import KNeighborsClassifier
import pickle #used to save and load the data
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix 

with open(r'saved_databases/credit.pkl','rb') as f:
    x_credit_training, y_credit_training, x_credit_test, y_credit_test = pickle.load(f)
    

KNN_credit = KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2)
KNN_credit.fit(x_credit_training,y_credit_training)

previsions = KNN_credit.predict(x_credit_test)

accuracy = accuracy_score(y_credit_test,previsions)

print(classification_report(y_credit_test,previsions))

#Creating a confusion matrix of the data
cm = ConfusionMatrix(KNN_credit)
cm.fit(x_credit_training,y_credit_training)
cm.score(x_credit_test,y_credit_test)