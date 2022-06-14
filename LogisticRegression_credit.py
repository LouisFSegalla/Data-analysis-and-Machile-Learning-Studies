# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 09:34:19 2022

@author: luisf
"""

import pickle #used to save and load the data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix 


with open(r'saved_databases/credit.pkl','rb') as f:
    x_credit_training, y_credit_training, x_credit_test, y_credit_test = pickle.load(f)
    
    
LogisticCredit = LogisticRegression(random_state=1)

LogisticCredit.fit(x_credit_training,y_credit_training)

predictions = LogisticCredit.predict(x_credit_test)

print('accuracy_score = ', accuracy_score(y_credit_test,predictions))

#Creating a confusion matrix of the data
cm = ConfusionMatrix(LogisticCredit)
cm.fit(x_credit_training,y_credit_training)
cm.score(x_credit_test, y_credit_test)

print(classification_report(y_credit_test,predictions))