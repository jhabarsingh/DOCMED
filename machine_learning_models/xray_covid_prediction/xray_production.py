'''
Detecting COVID-19 with Chest X Ray using PyTorch
Image classification of Chest X Rays in one of three classes: Normal, Viral Pneumonia, COVID-19
'''

import os
import pickle
import shutil
import random
import torch
import torchvision
import numpy as np
from PIL import Image

torch.manual_seed(0)
print('Using PyTorch version', torch.__version__)

#new_obj.resample('M').sum().plot(kind="bar")
#plt.show()



# Preparing Training and Test Sets

class_names = ['normal', 'viral', 'covid']
root_dir = 'COVID-19 Radiography Database'
source_dirs = ['NORMAL', 'Viral Pneumonia', 'COVID-19']

# Creating Custom Dataset

def joiner(file_name):
    '''Returns File location used in the django backend
    '''
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)


class ChestXRayDataset(torch.utils.data.Dataset):
    """Create dataset -> ChestXRayDataset"""
    def __init__(self, image_dirs, transform):
        def get_images(class_name):
            path = os.path.dirname(os.path.abspath(__file__))
            print(path, 'aaaaaaaaaaaaa')
            images = [x for x in os.listdir(joiner(image_dirs[class_name])) \
                     if x.lower().endswith('png')]
            
            print(f'Found {len(images)} {class_name} examples')
            return images

        self.images = {}
        self.class_names = ['normal', 'viral', 'covid']

        for c in self.class_names:
            self.images[c] = get_images(c)

        self.image_dirs = image_dirs
        self.transform = transform

    def __len__(self):
        # Total no of images
        return sum([len(self.images[c]) for c in self.class_names])

    def __getitem__(self, index):
        # Selects an image randomly choosen returns image [Tensor]
        class_name = random.choice(self.class_names)
        index = index % len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.class_names.index(class_name)


# Image Transformations


test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(244, 244)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
])

# Prepare DataLoader

test_dirs = {
    'normal': joiner('COVID-19 Radiography Database/test/normal'),
    'viral': joiner('COVID-19 Radiography Database/test/viral'),
    'covid': joiner('COVID-19 Radiography Database/test/covid'),
}

test_dataset = ChestXRayDataset(test_dirs, test_transform) 
batch_size = 1
dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                      shuffle=True)



resnet18 = None

with open(joiner('pick_resnet18.obj'), 'rb') as rfile:
    resnet18 = pickle.load(rfile)


# Creating the Model

def result(images, labels, preds):
    '''Predict if a patient is corona positive or  not
    '''
    image = images[0]
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0., 1.)
    return f'{class_names[int(preds[0].numpy())]}'
    

def show_preds():
    resnet18.eval()
    images, labels = next(iter(dl_test))
    outputs = resnet18(images)
    _, preds = torch.max(outputs, 1)
    show_images(images, labels, preds)


# Training the Model


# Final Results

def predict(image):
    resnet18.eval()
    images, labels = next(iter(dl_test))
    image = image
    images[0] = test_transform(image)
    print(np.ndim(images))
    outputs = resnet18(images)
    _, preds = torch.max(outputs, 1)
    return result(images, labels, preds)
