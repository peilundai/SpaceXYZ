
# coding: utf-8

# In[1]:


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

import spacexyz
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

K.tensorflow_backend._get_available_gpus()


# ## Prepare dataset

# In[2]:


# train_image_path = "/scratch2/peilun/train_images/"
# train_label_path = "/scratch2/peilun/train_labels/"
# val_image_path = "/scratch2/peilun/val_images/"
# val_label_path = "/scratch2/peilun/val_labels/"

train_image_path = "/scratch2/peilun/originalImages_augmented"
train_label_path = "/scratch2/peilun/GT_augmented"
val_image_path = "/scratch2/peilun/val_types/"
val_label_path = "/scratch2/peilun/val_types_labels/"

train_images = spacexyz.path2filelist(train_image_path)
train_labels = spacexyz.path2filelist(train_label_path)
val_images = spacexyz.path2filelist(val_image_path)
val_labels = spacexyz.path2filelist(val_label_path)

assert(len(train_images) == len(train_labels))
assert(len(val_images) == len(val_labels))

n_training = len(train_images)
n_val = len(val_images)
print(n_training)
print(n_val)

input_size = (512, 512, 3)
output_size = (512, 512)

n_classes = 1+7

# initialize data
X_train = np.zeros([n_training, *input_size]).astype(np.uint8)
y_train = np.zeros([n_training, *output_size]).astype(np.uint8)

X_val = np.zeros([n_val, *input_size]).astype(np.uint8)
y_val = np.zeros([n_val, *output_size]).astype(np.uint8)


# In[3]:


####################################################
############# Read in training dataset #############
####################################################

print("reading in ", n_training, " training samples...")
for i in range(n_training):
    print(i, end='.')
    t_image = cv2.imread(join(train_image_path, train_images[i]))
    t_label = cv2.imread(join(train_label_path, train_labels[i]))
    X_train[i,:,:,:] = cv2.resize(t_image, input_size[:2])
    y_train[i,:,:] = cv2.resize(t_label[:,:,0], output_size[:2], interpolation=cv2.INTER_NEAREST)


# In[4]:


y_train = keras.utils.to_categorical(y_train, num_classes=n_classes, dtype='float32')


# In[5]:


####################################################
############# Read in validation dataset ###########
####################################################

print("reading in ", n_val, " eval samples...")
for i in range(n_val):
    print(i,end= '.')
    v_image = cv2.imread(join(val_image_path, val_images[i]))
    v_label = cv2.imread(join(val_label_path, val_labels[i]))
    X_val[i,:,:,:] = cv2.resize(v_image, input_size[:2])
    y_val[i,:,:] = cv2.resize(v_label[:,:,0], output_size[:2], interpolation=cv2.INTER_NEAREST)
y_val = keras.utils.to_categorical(y_val, num_classes=n_classes, dtype='float32')


# ## Model training

# In[6]:


####################################################
############# Preprocess data ######################
####################################################

preprocessing_fn = get_preprocessing('resnet34')
x = preprocessing_fn(X_train)


# In[15]:


class TestCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


# In[16]:


####################################################
############# Set model parameters #################
####################################################

model = Unet(backbone_name='resnet34', classes=n_classes, activation='softmax')
model.compile('Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


# In[18]:


####################################################
############# Training model #######################
####################################################

for i in range(10):
    print('in iteration: ', i)
    model.fit(x, y_train,  validation_data=(X_val, y_val), callbacks=[TestCallback((X_val, y_val))], batch_size=1, epochs=10, verbose=True)
    model_name = 'Unet_epoch_'+str(10*(i+1))+'.h5'
    model.save(model_name)


# ## Model validation

# In[67]:


# pred = model.predict(X_val, batch_size=None, verbose=1, steps=None)


# ## Visualize result

# In[68]:


# import imgaug as ia
# from imgaug import augmenters as iaa

# one_hot = spacexyz.onehot2ind(pred)

# k=19
# label = one_hot[k,:,:]
# segmap = label.astype(np.int32)
# segmap = ia.SegmentationMapOnImage(segmap, shape=(512, 512), nb_classes=1+6)
# plt.imshow(segmap.draw_on_image(X_val[k,:,:,:]))
# cv2.imwrite('messigray.png',segmap.draw_on_image(X_val[k,:,:,:]))

