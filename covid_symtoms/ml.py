import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler , Normalizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


corona_pd=pd.read_csv("../input/covid19-symptoms-checker/Cleaned-Data.csv")
corona_pd.sample(5)

#Returns the  meta data of the dataset.
corona_pd.info()


#Returns the information like mean,max,min,etc., of the dataset.
corona_pd.describe()


