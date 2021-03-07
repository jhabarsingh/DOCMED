## COVID SYMPTOMS

**OBJECTIVE** : Here, we are predicting the Covid possibility on the accordance of
symptoms given by the user. For this we tried two of the ML models K-means and
RandomforestClassifier.

**INPUT FORMAT** :
From the user we are taking the following features :
* Fever
* Tiredness
* Dry-Cough
* Difficulty-in-Breathing
* Sore-Throat
* Pains
* Nasal-Congestion
* Runny-Nose
* Diarrhea
* Age
* Gender <br />
In the process of Data Preprocessing we Encoded all the Normal form raw-data into
Numerical form with the help of Label Encoder and classifying the data on the basis
of Severity, Age and Gender into No-risk, Low-risk, Moderate-risk or High-risk for the
Covid19.

**OUTPUT FORMAT** : The output format of our model will be in one of the four
cluster i.e. No-risk, Low-risk, Moderate-risk or High-risk for the Covid19. We have
trained our model to classify any suitable data into one of this cluster.

**WHY THIS MODEL** : The foremost
reason for choosing RandomForest
Classifier as our model to classify the input into one of the four specified clusters is
its highest accuracy among others.
It decorrelates the trees with the introduction of splitting on a random subset of
features. This means that at each split of the tree, the model considers only a small
subset of features rather than all of the features of the model.
By tuning the hyperparameters we achieved two goals,
  1. It increases the predictive power of the model
  2. Makes our model Faster.

