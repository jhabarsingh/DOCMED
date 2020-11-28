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
import os
warnings.filterwarnings('ignore')


def joiner(file_name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)

from sklearn.ensemble import RandomForestClassifier

def predict(a):
	with open(joiner('data.obj'), 'rb') as rfile:
		binary_file = pickle.load(rfile)
	rfc1 = binary_file
	import random
	pred=rfc1.predict([a])
	return pred






