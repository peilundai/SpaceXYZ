from os import listdir
from os.path import isfile, join

import numpy as np
import cv2
import matplotlib.pyplot as plt

from segmentation_models import PSPNet
from segmentation_models import FPN
from segmentation_models import Unet
from segmentation_models.segmentation_models.backbones import get_preprocessing

from keras import backend as K
import keras

def hello_world():
    print("hello, world")

def path2filelist(path):
    images = [f for f in listdir(path) if isfile(join(path, f))]
    
    if '.DS_Store' in images:
        images.remove('.DS_Store')
        images.sort()
    
    return images

def onehot2ind(one_hot):
    argmax = np.argmax(one_hot, axis=-1)
    return argmax