## X-RAY CLASSIFIER

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

