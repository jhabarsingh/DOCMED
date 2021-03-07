## X-RAY PREDICTION

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