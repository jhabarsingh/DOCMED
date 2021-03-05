### X-RAY CLASSIFIER

**OBJECTIVE** : Using Keras 2D Convolution Model, we create a X-RAY Classifier
that predicts whether the input image is an X-RAY report or not.

**INPUT FORMAT** : We have labelled our input data into two categories
that is covid and non-covid. Input data is a BGR format image.

1. **DATA MINING** : The input data is a BGR format image and the dimensions of
every image a different therefore we have used OPEN-CV to first convert our
image format from BGR to RGB and then to resize all the images to a unique
dimension of (224, 224). Finally changed the datatype of the image from List
to numpy array so that we could perform the mathematical operations on the
data provided by the numpy library.

2. **DATA SPLITTING** : We have splitted our dataset into two:
    1. Train dataset
    2. Test data set
    * Ratio is 7 : 3 respectively

3. **NORMALIZATION** : The dataset is a 2D array (numpy dataset) and it
contains integers in the range [0, 225]. So when we would perform
mathematical operations on the given array the number range would increase
even further which Would take extra space and even time complexity to
perform mathematical operations on these numbers would increase. To
overcome this problem we have decreased the range of the integer to [0, 1].
I.e divided the numbers by 225.

**OUTPUT FORMAT** : It returns an array of booleans of size equal to the test
dataset size and the value of these booleans depicts whether the inputted image is
of chest X-RAY or not.

**WHY THIS MODEL** : We used two models to train our dataset
1. Pytorch ResNet18
2. Tensorflow InceptionResNetV2
* Out of these two we had chosen InceptionResNetV2 because of the higher
accuracy.


### CT SCAN PREDICTION

**OBJECTIVE** : Using PyTorch, we create a COVID-19 classifier that predicts
whether a patient is suffering from coronavirus or not, using chest CT scans of
different patients.

**INPUT FORMAT**
1. **About Data** <br/>
We have the positive class with the scans of COVID-19 positive patients, whereas
the negative class contains a mixture of healthy patients, and patients suffering from
other (non-COVID-19) diseases that may cause opacities in the lungs. In order to
train a robust classifier, we must have the information about the non-COVID-19
patients as well.
The dataset is divided into three categories: the train set, validation set, and the test
set. The data folder contains two categories Covid and Non-Covid in which the
above three splits are present. We write a function to read these files and put them
into a list of strings.

2. **Input pre-processing and data augmentation** <br />
For the training data:
    * Resize the shorter side of the image to 256 while maintaining the aspect
ratio.
    * Do a random crop of size ranging from 50% to 100% of the dimensions
of the image, and aspect ratio ranging randomly from 75% to 133% of
the original aspect ratio.
    * Finally, the crop is resized to 224 × 224.
    * We use a mini-batch size of 10.
    * Horizontally flip the image with a probability of 0.
    * Normalize the image to have 0 mean and standard deviation of 1.
For testing:
    * Resize the image to 224 × 224.
    * Normalize the image to have mean 0 and standard deviation of 
       
**OUTPUT FORMAT** : We then create the COVIDCT Dataset class which basically
subclasses the torch.utils.data.Dataset class. The dataset returns a dictionary
containing the image tensor, the label tensor, and a list of image paths included in
the batch. The Output contains images classified as covid or non-covid.


**WHY THIS MODEL** : We used the Convolutional Neural Network Architecture
models like -
1. ResNet18
2. InceptionResNetV2
3. Dense169
4. efficientNet
We have used CNN Architecture because --
* They have very good feature learning capabilities at different levels
than a Conventional Neural Network.
* They have a very good feature of weight sharing, which reduces the
number of parameters in our model. For e.g. we tried to implement a 1
layered neural network with 250 neurons and we got our parameter
size as 19601 ( for MNIST data : 250*784+1 = 19601 ). But when the
same network was implemented using CNN our parameter size was
reduced to 260 ( more than 75 times !!! ).
* They are also very good feature extractors which helped us in our
pre-training process.
* In terms of performance the CNN out performed the NN.

### COVID SYMPTOMS

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


### X-RAY PREDICTION

**OBJECTIVE** : Using PyTorch Library, we have created a COVID-19 classifier that
predicts whether a patient is suffering from coronavirus or not, using chest X-ray of
any individual.

**INPUT FORMAT** :
1. **About Data**
We have the positive class with the X-ray of COVID-19 positive patients, whereas
the negative class contains a mixture of healthy patients, and patients suffering from
other (non-COVID-19) diseases that may cause opacities in the lungs. In order to
train a robust classifier, we must have the information about the non-COVID-19
patients as well.
The dataset is divided into three categories: the train set, validation set, and the test
set. The data folder contains two categories Covid and Non-Covid in which the
above three splits are present. We write a function to read these files and put them
into a list of strings.

2. **Input pre-processing and data augmentation** <br />
For the training data:
  * Resize the shorter side of the image to 244*244 while maintaining the
aspect ratio.
  * Conversion of image to Tensors using Torchvision transform.
  * We use a mini-batch size of 6.
  * Normalize the image to have 0 mean and standard deviation of 1.
For testing:
  * Resize the image to 244 × 244.
  * Normalize the image to have mean 0 and standard deviation of 1.

**OUTPUT FORMAT** : We then create the dl_train and dl_test Dataset class
which basically subclasses the torch.utils.data.Dataset class. The dataset returns a
dictionary containing the image tensor, the label tensor, and a list of image paths
included in the batch. The Output contains images classified as covid or non-covid.

**WHY THIS MODEL** : we used two models to train our dataset
1. Pytorch ResNet18
2. Tensorflow InceptionResNetV2
Out of these two we had chosen ResNet18 because
1. Higher Accuracy.
2. Weight changes are easily adjustable.
3. Loss in our model was very less which signifies the better model compared to
others.
4. Resnet18 also provides very good feature extractors which helped us in our
pre-training process.
