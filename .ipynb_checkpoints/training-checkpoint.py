
# coding: utf-8

# In[16]:


from os import listdir
from os.path import isfile, join

import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage

from segmentation_models import PSPNet
from segmentation_models import FPN
from segmentation_models import Unet
from segmentation_models import Linknet
from segmentation_models.segmentation_models.backbones import get_preprocessing

from keras import backend as K
import keras

import spacexyz
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

K.tensorflow_backend._get_available_gpus()


# ## Set Parameters

# In[17]:


train_image_path = "/scratch2/peilun/relabeled_data/new_train_images/"
train_label_path = "/scratch2/peilun/relabeled_data/new_train_labels"
val_image_path = "/scratch2/peilun/relabeled_data/new_val_images"
val_label_path = "/scratch2/peilun/relabeled_data/new_val_labels"

# train_image_path = "/scratch2/peilun/train_images"
# train_label_path = "/scratch2/peilun/train_labels"
# val_image_path = "/scratch2/peilun/val_images"
# val_label_path = "/scratch2/peilun/val_labels"

n_classes = 1+7

input_size = (384, 384, 3)
output_size = (384, 384)

# Unet, PSPNet, FPN, Linknet
model_name = 'PSPNet'

# vgg16, vgg19, resnet18, resnet34, resnet50, resnet101, resnet152, resnext50, resnext101, densenet121, densenet169, densenet201, inceptionv3, inceptionresnetv2
backbone_name = 'resnet34'

# number of 10 epoches between saving models
n_save = 2
epc_num = 10

# TODO to resize back the image


# ## Prepare dataset

# In[18]:


####################################################
############# Read in files ########################
####################################################

train_images = spacexyz.path2filelist(train_image_path)
train_labels = spacexyz.path2filelist(train_label_path)
val_images = spacexyz.path2filelist(val_image_path)
val_labels = spacexyz.path2filelist(val_label_path)

assert(len(train_images) == len(train_labels))
assert(len(val_images) == len(val_labels))

n_training = len(train_images)
n_val = len(val_images)
print("Number of training images: ", n_training)
print("Number of validation or testing images: ", n_val)

# initialize data
X_train = np.zeros([n_training, *input_size]).astype(np.uint8)
y_train = np.zeros([n_training, *output_size]).astype(np.uint8)

X_val = np.zeros([n_val, *input_size]).astype(np.uint8)
y_val = np.zeros([n_val, *output_size]).astype(np.uint8)


####################################################
############# Read in training dataset #############
####################################################

train_size = np.zeros([n_training, 2])
val_size = np.zeros([n_val, 2])

print("reading in ", n_training, " training samples...")
for i in range(n_training):
    print(i, end='.')
    t_image = cv2.imread(join(train_image_path, train_images[i]))
    t_label = cv2.imread(join(train_label_path, train_labels[i]))
    print(t_image.shape[:2])
    train_size[i,:] = t_image.shape[:2]
    X_train[i,:,:,:] = cv2.resize(t_image, input_size[:2])
    y_train[i,:,:] = cv2.resize(t_label[:,:,0], output_size[:2], interpolation=cv2.INTER_NEAREST)

    
y_train = keras.utils.to_categorical(y_train, num_classes=n_classes, dtype='float32')


####################################################
############# Read in validation dataset ###########
####################################################

print("reading in ", n_val, " eval samples...")
for i in range(n_val):
    print(i,end= '.')
    v_image = cv2.imread(join(val_image_path, val_images[i]))
    v_label = cv2.imread(join(val_label_path, val_labels[i]))
    val_size[i,:] = v_image.shape[:2]
    X_val[i,:,:,:] = cv2.resize(v_image, input_size[:2])
    y_val[i,:,:] = cv2.resize(v_label[:,:,0], output_size[:2], interpolation=cv2.INTER_NEAREST)
y_val = keras.utils.to_categorical(y_val, num_classes=n_classes, dtype='float32')


# ## Model training

# In[ ]:


####################################################
############# Preprocess data ######################
####################################################

preprocessing_fn = get_preprocessing('resnet34')
x = preprocessing_fn(X_train)

## callback function to evaluate test data at the end of each training epoch

class TestCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

####################################################
############# Set model parameters #################
####################################################

if model_name == 'Unet':
    model = Unet(backbone_name=backbone_name, classes=n_classes, activation='softmax')
elif model_name == 'PSPNet':
    model = PSPNet(backbone_name=backbone_name, classes=n_classes, activation='softmax')
elif model_name == 'FPN':
    model = FPN(backbone_name=backbone_name, classes=n_classes, activation='softmax')
elif model_name == 'Linknet':
    model = Linknet(backbone_name=backbone_name, classes=n_classes, activation='softmax')
else:
    print('Please provide the right model name')

model.compile('Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy']) 


####################################################
############# Training model #######################
####################################################

for i in range(n_save):
    print('==============================')
    print('in iteration: ', i+1)
    print('==============================')
    
    model.fit(x, y_train,  validation_data=(X_val, y_val), callbacks=[TestCallback((X_val, y_val))], batch_size=1, epochs=epc_num, verbose=True)
    mname = 'PSPNet_relabeled_'+str(epc_num*(i+1))+'.h5'
    
    print('==============================')
    print('saving model after epoch ', (i+1)*epc_num)
    print('==============================')
    
#     save_name = model_name+"_noaug_epoch_"+str(10*(i+1))+".h5"
    model.save(mname)


# ## Result visualization

# In[5]:


# model = keras.models.load_model('./Unet_epoch_90.h5_noaug_epoch_90.h5')


# In[7]:


# pred = model.predict(X_val, batch_size=None, verbose=1, steps=None)


# In[ ]:


# # resize to original

# k = 12
# p = pred[k,:,:]
# p_original = skimage.transform.resize(p, val_size[k,:], order=0, mode='reflect', cval=0)
# label = spacexyz.onehot2ind(p_original)
# segmap = label.astype(np.int32)
# segmap = ia.SegmentationMapOnImage(segmap, shape=val_size[k,:], nb_classes=n_classes)
# X_or = cv2.imread(join(train_image_path, train_images[k]))
# plt.imshow(segmap.draw_on_image(X_or))
# # cv2.imwrite('messigray.png',segmap.draw_on_image(X_val[k,:,:,:]))


# In[ ]:


# import imgaug as ia
# from imgaug import augmenters as iaa

# one_hot = spacexyz.onehot2ind(pred)

# k=10
# label = one_hot[k,:,:]
# segmap = label.astype(np.int32)
# segmap = ia.SegmentationMapOnImage(segmap, shape=(1024, 1024), nb_classes=1+6)
# plt.imshow(segmap.draw_on_image(X_val[k,:,:,:]))
# cv2.imwrite('messigray.png',segmap.draw_on_image(X_val[k,:,:,:]))

