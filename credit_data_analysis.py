# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 12:56:16 2022

@author: luisf
"""

#importing the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

#loading the database
base_credit = pd.read_csv(r'archive/credit_data.csv')

#general information about the dataset
info = base_credit.describe()
print(info)

#number of people who paied their loan
num_paied = np.unique(base_credit['default'],return_counts=True)

#ploting the data 
plt.hist(x = base_credit['default'])
plt.show()
plt.hist(x = base_credit['age'])
plt.show()
plt.hist(x = base_credit['income'])
plt.show()
plt.hist(x = base_credit['loan'])
plt.show()

###################################
#needed this import to be able to open the plotly graphs in spyder (windows 10)
import plotly.io as pio
pio.renderers.default='browser'
##################################
#dynamic plot using plotly
# graph = px.scatter_matrix(base_credit,dimensions=['age','income'],color=['default'])
# graph.show()

info_clients_negative = base_credit[base_credit['age'] < 0]

#erasing the lines with negative age
base_credit_non_negative = base_credit.drop(base_credit[base_credit['age'] < 0].index)

#filll the negative values of age with the mean age of the dataset
base_credit_filled = base_credit
base_credit_filled.loc[base_credit_filled['age']<0,'age'] = base_credit_non_negative['age'].mean()