# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 08:47:30 2022

@author: luisf
"""

#importing the necessary libraries
import pandas as pd
import numpy as np
import pickle #used to save the data
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from yellowbrick.classifier import ConfusionMatrix 


with open(r'saved_databases/credit.pkl','rb') as f:
    x_credit_training, y_credit_training, x_credit_test, y_credit_test = pickle.load(f)
    
naive_credit_data = GaussianNB()
naive_credit_data.fit(x_credit_training, y_credit_training)

previsions_tests = naive_credit_data.predict(x_credit_test)

accuracy = accuracy_score(y_credit_test,previsions_tests)
confusion_matrix_output = confusion_matrix(y_credit_test,previsions_tests)
print(classification_report(y_credit_test, previsions_tests))
