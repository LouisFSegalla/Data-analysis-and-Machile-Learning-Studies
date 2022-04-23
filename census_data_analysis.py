# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 09:18:11 2022

@author: luisf
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler

#This base has numerical and categoric atributes
base_census = pd.read_csv(r'archive/census.csv')

# graph = px.treemap(base_census, path=['workclass','age'])
# graph.show()

graph = px.parallel_categories(base_census,dimensions=['occupation','income'])
graph.show()