import pickle
import pandas as pd
import numpy as np
#import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler , Normalizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import norm
from scipy import stats
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


from sklearn.ensemble import RandomForestClassifier
rfc1=RandomForestClassifier(criterion= 'gini', max_depth= 4, max_features= 'sqrt', n_estimators= 100)
binary_file = open('data.obj', 'rb')
rfc1 = pickle.load(binary_file)
binary_file.close()

import random

a = [0, 1, 1, 1, 0]
for i in range(5, 23):
    a.append(1)
pred = rfc1.predict([a])
# print(pred)


