# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:58:55 2022

@author: luisf
"""

from sklearn.neural_network import MLPClassifier
import pickle #used to save and load the data
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix 


with open(r'saved_databases/credit.pkl','rb') as f:
    x_credit_training, y_credit_training, x_credit_test, y_credit_test = pickle.load(f)
    

neural_network_credit = MLPClassifier(activation='relu',max_iter=2000,verbose=True,solver='adam',tol=1e-5,hidden_layer_sizes=(50,50))
neural_network_credit.fit(x_credit_training,y_credit_training)

predictions = neural_network_credit.predict(x_credit_test)

print('accuracy_score = ', accuracy_score(y_credit_test,predictions))

#Creating a confusion matrix of the data
cm = ConfusionMatrix(neural_network_credit)
cm.fit(x_credit_training,y_credit_training)
cm.score(x_credit_test, y_credit_test)

print(classification_report(y_credit_test,predictions))