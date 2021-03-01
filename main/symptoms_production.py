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

from sklearn.ensemble import RandomForestClassifier

def joiner(folder_name, file_name):
	paths = os.path.dirname(os.path.abspath(__file__))
	paths = os.path.dirname(paths)
	paths = os.path.join(paths, folder_name)
	paths = os.path.join(paths, file_name)
	return paths


def predict_covid_from_symptoms(a):
	with open(joiner('machine_learning_models/symptoms_covid_prediction', 'pickled_model.obj'), 'rb') as rfile:
		binary_file = pickle.load(rfile)
	rfc1 = binary_file
	import random
	pred=rfc1.predict([a])
	return pred






