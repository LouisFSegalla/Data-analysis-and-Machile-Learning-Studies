# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 09:18:11 2022

@author: luisf
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle #used to save the data
#This base has numerical and categoric atributes
base_census = pd.read_csv(r'archive/census.csv')

# Using some of the specific plots of the plotly library
# graph = px.treemap(base_census, path=['workclass','age'])
# graph.show()

# graph = px.parallel_categories(base_census,dimensions=['occupation','income'])
# graph.show()

x_census = base_census.iloc[:,0:14].values
y_census = base_census.iloc[:,14].values

#Transforming all the categoric atributes to numerical values so calculations can be done
labelEncoder_workclass = LabelEncoder()
labelEncoder_education = LabelEncoder()
labelEncoder_marital_status = LabelEncoder()
labelEncoder_occupation = LabelEncoder()
labelEncoder_relationship = LabelEncoder()
labelEncoder_race = LabelEncoder()
labelEncoder_sex= LabelEncoder()
labelEncoder_country= LabelEncoder()

#####################################################################################
# When dealing with categoric atributes it's better to first use the label encoder to transform them
# into a numerical integer value. Them use the OneHotEncoder to mask this numerical data into collums
# Thats what's being made in lines 45 to 62
#####################################################################################

#Rewriting the values of the atributes as numbers
x_census[:,1]  = labelEncoder_workclass.fit_transform(x_census[:,1])
x_census[:,3]  = labelEncoder_education.fit_transform(x_census[:,3])
x_census[:,5]  = labelEncoder_marital_status.fit_transform(x_census[:,5])
x_census[:,6]  = labelEncoder_occupation.fit_transform(x_census[:,6])
x_census[:,7]  = labelEncoder_relationship.fit_transform(x_census[:,7])
x_census[:,8]  = labelEncoder_race.fit_transform(x_census[:,8])
x_census[:,9]  = labelEncoder_sex.fit_transform(x_census[:,9])
x_census[:,13] = labelEncoder_country.fit_transform(x_census[:,13])


#Using the OneHotEncoder import to transform the data
OneHotEncoder_census = ColumnTransformer(transformers=[('OneHot',OneHotEncoder(),[1,3,5,6,7,8,9,13])], remainder='passthrough')
x_census = OneHotEncoder_census.fit_transform(x_census).toarray()

#Scaling all the atributes
scaller_sensus = StandardScaler()
x_census = scaller_sensus.fit_transform(x_census)

###############################################################################################
#Separating the data into training and test datasets 
###############################################################################################

x_census_training, x_census_test, y_census_training, y_census_test = train_test_split(x_census,y_census,test_size=0.15,random_state=0)

with open('saved_databases/census.pkl',mode='wb') as f:
    pickle.dump([x_census_training, y_census_training, x_census_test, y_census_test],f)