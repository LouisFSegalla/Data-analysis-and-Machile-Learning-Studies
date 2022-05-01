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

with open(r'saved_databases/credit.pkl','rb') as f:
    x_credit_training, y_credit_training, x_credit_test, y_credit_test = pickle.load(f)
    
    
randForestCredit = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)
randForestCredit.fit(x_credit_training,y_credit_training)

predictions = randForestCredit.predict(x_credit_test)

accuracy = accuracy_score(y_credit_test,predictions)
confusion_matrix_output = confusion_matrix(y_credit_test,predictions)
print(classification_report(y_credit_test, predictions))

#Creating a confusion matrix of the data
cm = ConfusionMatrix(randForestCredit)
cm.fit(x_credit_training,y_credit_training)
cm.score(x_credit_test, y_credit_test)
