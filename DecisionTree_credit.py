# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 19:28:31 2022

@author: luisf
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree 
import matplotlib.pyplot as plt
import pickle #used to save and load the data
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix 

with open(r'saved_databases/credit.pkl','rb') as f:
    x_credit_training, y_credit_training, x_credit_test, y_credit_test = pickle.load(f)
    

#creating the decision tree object
creditTree = DecisionTreeClassifier(criterion='entropy',random_state=0)
#training the tree with the dataset
creditTree.fit(x_credit_training,y_credit_training)

#Using our training dataset with the trained tree
previsions = creditTree.predict(x_credit_test)

#Accuracy test
accuracy = accuracy_score(y_credit_test,previsions)

#Creating a confusion matrix of the data
cm = ConfusionMatrix(creditTree)
cm.fit(x_credit_training,y_credit_training)
cm.score(x_credit_test,y_credit_test)

#Some more info about the trained tree
print(classification_report(y_credit_test,previsions))

#showing the tree
previsors_labels=['income','age','loan']
fig,axes = plt.subplots(nrows=1,ncols=1,figsize=(20,20))
tree.plot_tree(creditTree,feature_names=previsors_labels,class_names=['pay', 'does not pay'],filled=True)
fig.savefig(r'saved_databases/CreditTree.png')