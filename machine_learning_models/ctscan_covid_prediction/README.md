## CT SCAN PREDICTION

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

