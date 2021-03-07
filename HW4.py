#!/usr/bin/env python
# coding: utf-8

# # HW: X-ray images classification
# --------------------------------------

# Before you begin, open Mobaxterm and connect to triton with the user and password you were give with. Activate the environment `2ndPaper` and then type the command `pip install scikit-image`.

# In this assignment you will be dealing with classification of 32X32 X-ray images of the chest. The image can be classified into one of four options: lungs (l), clavicles (c), and heart (h) and background (b). Even though those labels are dependent, we will treat this task as multiclass and not as multilabel. The dataset for this assignment is located on a shared folder on triton (`/MLdata/MLcourse/X_ray/'`).

# In[ ]:


import os
import numpy as np
from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D, Dropout
from tensorflow.keras.layers import Flatten, InputLayer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import *

from tensorflow.keras.initializers import Constant
from tensorflow.keras.datasets import fashion_mnist
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from skimage.io import imread

from skimage.transform import rescale, resize, downscale_local_mean
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# In[ ]:


import tensorflow as tf

config = tf.compat.v1.ConfigProto(gpu_options=
                                  tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                  # device_count = {'GPU': 1}
                                  )
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


# In[ ]:


def preprocess(datapath):
    # This part reads the images
    classes = ['b', 'c', 'l', 'h']
    imagelist = [fn for fn in os.listdir(datapath)]
    N = len(imagelist)
    num_classes = len(classes)
    images = np.zeros((N, 32, 32, 1))
    Y = np.zeros((N, num_classes))
    ii = 0
    for fn in imagelist:

        src = imread(os.path.join(datapath, fn), 1)
        img = resize(src, (32, 32), order=3)

        images[ii, :, :, 0] = img
        cc = -1
        for cl in range(len(classes)):
            if fn[-5] == classes[cl]:
                cc = cl
        Y[ii, cc] = 1
        ii += 1

    BaseImages = images
    BaseY = Y
    return BaseImages, BaseY


# In[ ]:


def preprocess_train_and_val(datapath):
    # This part reads the images
    classes = ['b', 'c', 'l', 'h']
    imagelist = [fn for fn in os.listdir(datapath)]
    N = len(imagelist)
    num_classes = len(classes)
    images = np.zeros((N, 32, 32, 1))
    Y = np.zeros((N, num_classes))
    ii = 0
    for fn in imagelist:

        images[ii, :, :, 0] = imread(os.path.join(datapath, fn), 1)
        cc = -1
        for cl in range(len(classes)):
            if fn[-5] == classes[cl]:
                cc = cl
        Y[ii, cc] = 1
        ii += 1

    return images, Y


# In[ ]:


# Loading the data for training and validation:
src_data = '/MLdata/MLcourse/X_ray/'
train_path = src_data + 'train'
val_path = src_data + 'validation'
test_path = src_data + 'test'
BaseX_train, BaseY_train = preprocess_train_and_val(train_path)
BaseX_val, BaseY_val = preprocess_train_and_val(val_path)
X_test, Y_test = preprocess(test_path)

# In[ ]:


keras.backend.clear_session()

# ### PART 1: Fully connected layers
# --------------------------------------

# ---
# <span style="color:red">***Task 1:***</span> *NN with fully connected layers.
#
# Elaborate a NN with 2 hidden fully connected layers with 300, 150 neurons and 4 neurons for classification. Use ReLU activation functions for the hidden layers and He_normal for initialization. Don't forget to flatten your image before feedforward to the first dense layer. Name the model `model_relu`.*
#
# ---

# In[ ]:


# --------------------------Impelment your code here:-------------------------------------
n_hidden_start = 300
model_relu = Sequential()
model_relu.add(Dense(n_hidden_start, activation='relu', kernel_initializer='he_normal', input_shape=(32 * 32,)))
model_relu.add(Dense(int(n_hidden_start / 2), activation='relu', kernel_initializer='he_normal'))
model_relu.add(Dense(4, activation='softmax'))  # kernel_initializer= 'he_normal'
from tensorflow.keras.optimizers import Adam

# ----------------------------------------------------------------------------------------


# In[ ]:


model_relu.summary()

# In[ ]:


# Inputs:
input_shape = (32, 32, 1)
learn_rate = 1e-5
decay = 0
batch_size = 64
epochs = 25

# Define your optimizar parameters:
AdamOpt = Adam(lr=learn_rate, decay=decay)

# Compile the model with the optimizer above, accuracy metric and adequate loss for multiclass task. Train your model on the training set and evaluate the model on the testing set. Print the accuracy and loss over the testing set.

# In[ ]:


# --------------------------Impelment your code here:-------------------------------------
model_relu.compile(optimizer=AdamOpt, metrics=['accuracy'], loss='categorical_crossentropy')

train_images = BaseX_train.reshape((-1, 1024))
val_images = BaseX_val.reshape((-1, 1024))
test_images = X_test.reshape((-1, 1024))

history1 = model_relu.fit(train_images, BaseY_train, batch_size=batch_size, epochs=epochs,
                          validation_data=(val_images, BaseY_val), verbose=1)
loss_and_metrics1 = model_relu.evaluate(test_images, Y_test, verbose=1)

print("Task1: Test Loss is {:.2f} ".format(loss_and_metrics1[0]))
print("Task1: Test Accuracy is {:.2f} %".format(100 * loss_and_metrics1[1]))
# ----------------------------------------------------------------------------------------


# ---
# <span style="color:red">***Task 2:***</span> *Activation functions.*
#
# Change the activation functions to LeakyRelu or tanh or sigmoid. Name the new model `new_a_model`. Explain how it can affect the model.*
#
# ---

# In[ ]:


# --------------------------Impelment your code here:-------------------------------------
new_a_model = Sequential()
new_a_model.add(Dense(n_hidden_start, activation='tanh', kernel_initializer='he_normal', input_shape=(1024,)))
new_a_model.add(Dense(int(n_hidden_start / 2), activation='tanh', kernel_initializer='he_normal'))
new_a_model.add(Dense(4, activation='softmax'))  # kernel_initializer= 'he_normal'

new_a_model.compile(optimizer=AdamOpt, metrics=['accuracy'], loss='categorical_crossentropy')

history2 = new_a_model.fit(train_images, BaseY_train, batch_size=batch_size, epochs=epochs,
                           validation_data=(val_images, BaseY_val), verbose=1)
loss_and_metrics2 = new_a_model.evaluate(test_images, Y_test, verbose=1)

print("Task2: Test Loss is {:.2f} ".format(loss_and_metrics2[0]))
print("Task2: Test Accuracy is {:.2f} %".format(100 * loss_and_metrics2[1]))
# ----------------------------------------------------------------------------------------


# In[ ]:


new_a_model.summary()

# ---
# <span style="color:red">***Task 3:***</span> *Number of epochs.*
#
# Train the new model using 25 and 40 epochs. What difference does it makes in term of performance? Remember to save the compiled model for having initialized weights for every run as we did in tutorial 12. Evaluate each trained model on the test set*
#
# ---

# In[ ]:


# Inputs:
input_shape = (32, 32, 1)
learn_rate = 1e-5
decay = 0
batch_size = 64
epochs = 25

# Defining the optimizar parameters:
AdamOpt = Adam(lr=learn_rate, decay=decay)

# In[ ]:


# --------------------------Impelment your code here:-------------------------------------
model3a = Sequential()
model3a.add(Dense(n_hidden_start, activation='tanh', kernel_initializer='he_normal', input_shape=(1024,)))
model3a.add(Dense(int(n_hidden_start / 2), activation='tanh', kernel_initializer='he_normal'))
model3a.add(Dense(4, activation='softmax'))  # kernel_initializer= 'he_normal'

model3a.compile(optimizer=AdamOpt, metrics=['accuracy'], loss='categorical_crossentropy')

if not ("results" in os.listdir()):
    os.mkdir("results")
save_dir = "results/"
model_name = "weights_3a.h5"
model_path = os.path.join(save_dir, model_name)
model3a.save(model_path)
print('Saved trained model at %s ' % model_path)

history3a = model3a.fit(train_images, BaseY_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(val_images, BaseY_val), verbose=1)
loss_and_metrics3a = model3a.evaluate(test_images, Y_test, verbose=1)

print("Task3a: Test Loss is {:.2f} ".format(loss_and_metrics3a[0]))
print("Task3a: Test Accuracy is {:.2f} %".format(100 * loss_and_metrics3a[1]))

# -----------------------------------------------------------------------------------------


# In[ ]:


# Inputs:
input_shape = (32, 32, 1)
learn_rate = 1e-5
decay = 0
batch_size = 64
epochs = 40

# Defining the optimizar parameters:
AdamOpt = Adam(lr=learn_rate, decay=decay)

# In[ ]:


# --------------------------Impelment your code here:-------------------------------------
from tensorflow.keras.models import load_model

model3b = load_model("results/weights_3a.h5")

history3b = model3b.fit(train_images, BaseY_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(val_images, BaseY_val), verbose=1)
loss_and_metrics3b = model3b.evaluate(test_images, Y_test, verbose=1)

print("Task3b: Test Loss is {:.2f} ".format(loss_and_metrics3b[0]))
print("Task3b: Test Accuracy is {:.2f} %".format(100 * loss_and_metrics3b[1]))

# -----------------------------------------------------------------------------------------


# ---
# <span style="color:red">***Task 4:***</span> *Mini-batches.*
#
# Build the `model_relu` again and run it with a batch size of 32 instead of 64. What are the advantages of the mini-batch vs. SGD?*
#
# ---

# In[ ]:


keras.backend.clear_session()

# In[ ]:


# --------------------------Impelment your code here:-------------------------------------
model_relu2 = Sequential()
model_relu2.add(Dense(n_hidden_start, activation='relu', kernel_initializer='he_normal', input_shape=(1024,)))
model_relu2.add(Dense(int(n_hidden_start / 2), activation='relu', kernel_initializer='he_normal'))
model_relu2.add(Dense(4, activation='softmax'))  # kernel_initializer= 'he_normal'
# ----------------------------------------------------------------------------------------


# In[ ]:


batch_size = 32
epochs = 50

# Define your optimizar parameters:
AdamOpt = Adam(lr=learn_rate, decay=decay)

# In[ ]:


# --------------------------Impelment your code here:-------------------------------------
model_relu2.compile(optimizer=AdamOpt, metrics=['accuracy'], loss='categorical_crossentropy')

history4 = model_relu2.fit(train_images, BaseY_train, batch_size=batch_size, epochs=epochs,
                           validation_data=(val_images, BaseY_val), verbose=1)
loss_and_metrics4 = model_relu2.evaluate(test_images, Y_test, verbose=1)

print("Task4: Test Loss is {:.2f} ".format(loss_and_metrics4[0]))
print("Task4: Test Accuracy is {:.2f} %".format(100 * loss_and_metrics4[1]))
# ----------------------------------------------------------------------------------------


# ---
# <span style="color:red">***Task 4:***</span> *Batch normalization.*
#
# Build the `new_a_model` again and add batch normalization layers. How does it impact your results?*
#
# ---

# In[ ]:


keras.backend.clear_session()

# In[ ]:


# --------------------------Impelment your code here:-------------------------------------
new_a_model2 = Sequential()
new_a_model2.add(Dense(n_hidden_start, activation='tanh', kernel_initializer='he_normal', input_shape=(1024,)))
new_a_model2.add(BatchNormalization())
new_a_model2.add(Dense(int(n_hidden_start / 2), activation='tanh', kernel_initializer='he_normal'))
new_a_model2.add(BatchNormalization())
new_a_model2.add(Dense(4, activation='softmax'))  # kernel_initializer= 'he_normal'

# ---------------------------------------------------------------------------------------


# In[ ]:


batch_size = 32
epochs = 50

# Define your optimizar parameters:
AdamOpt = Adam(lr=learn_rate, decay=decay)
# Compile the network:


# In[ ]:


# Preforming the training by using fit
# --------------------------Impelment your code here:-------------------------------------
new_a_model2.compile(optimizer=AdamOpt, metrics=['accuracy'], loss='categorical_crossentropy')

history5 = new_a_model2.fit(train_images, BaseY_train, batch_size=batch_size, epochs=epochs,
                            validation_data=(val_images, BaseY_val), verbose=1)
loss_and_metrics5 = new_a_model2.evaluate(test_images, Y_test, verbose=1)

print("Task5: Test Loss is {:.2f} ".format(loss_and_metrics5[0]))
print("Task5: Test Accuracy is {:.2f} %".format(100 * loss_and_metrics5[1]))


# ----------------------------------------------------------------------------------------


# ### PART 2: Convolutional Neural Network (CNN)
# ------------------------------------------------------------------------------------

# ---
# <span style="color:red">***Task 1:***</span> *2D CNN.*
#
# Have a look at the model below and answer the following:
# * How many layers does it have?
# * How many filter in each layer?
# * Would the number of parmaters be similar to a fully connected NN?
# * Is this specific NN performing regularization?
#
# ---

# In[ ]:


def get_net(input_shape, drop, dropRate, reg):
    # Defining the network architecture:
    model = Sequential()
    model.add(Permute((1, 2, 3), input_shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='Conv2D_1',
                     kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', name='Conv2D_2',
                     kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', name='Conv2D_3',
                     kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='Conv2D_4',
                     kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='Conv2D_5',
                     kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # Fully connected network tail:
    model.add(Dense(512, activation='elu', name='FCN_1'))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(Dense(128, activation='elu', name='FCN_2'))
    model.add(Dense(4, activation='softmax', name='FCN_3'))
    model.summary()
    return model


# In[ ]:


input_shape = (32, 32, 1)
learn_rate = 1e-5
decay = 1e-03
batch_size = 64
epochs = 25
drop = True
dropRate = 0.3
reg = 1e-2
NNet = get_net(input_shape, drop, dropRate, reg)

# In[ ]:


NNet = get_net(input_shape, drop, dropRate, reg)

# In[ ]:


from tensorflow.keras.optimizers import *
import os
from tensorflow.keras.callbacks import *

# Defining the optimizar parameters:
AdamOpt = Adam(lr=learn_rate, decay=decay)

# Compile the network:
NNet.compile(optimizer=AdamOpt, metrics=['acc'], loss='categorical_crossentropy')

# Preforming the training by using fit
# IMPORTANT NOTE: This will take a few minutes!
h = NNet.fit(x=BaseX_train, y=BaseY_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0,
             validation_data=(BaseX_val, BaseY_val), shuffle=True)
# NNet.save(model_fn)


# In[ ]:


# NNet.load_weights('Weights_1.h5')


# In[ ]:


results = NNet.evaluate(X_test, Y_test)
print('test loss, test acc:', results)


# ---
# <span style="color:red">***Task 2:***</span> *Number of filters*
#
# Rebuild the function `get_net` to have as an input argument a list of number of filters in each layers, i.e. for the CNN defined above the input should have been `[64, 128, 128, 256, 256]`. Now train the model with the number of filters reduced by half. What were the results.
#
# ---

# In[ ]:


# --------------------------Impelment your code here:-------------------------------------
def get_net2(input_shape, drop, dropRate, reg, filters):
    # Defining the network architecture:
    model = Sequential()
    model.add(Permute((1, 2, 3), input_shape=input_shape))
    model.add(Conv2D(filters=filters[0], kernel_size=(3, 3), padding='same', activation='relu', name='Conv2D_1',
                     kernel_regularizer=regularizers.l2(reg)))

    for num in np.arange(1, len(filters) - 1, 2):
        if drop:
            model.add(Dropout(rate=dropRate))
        model.add(BatchNormalization(axis=1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=filters[num], kernel_size=(3, 3), padding='same', activation='relu',
                         name='Conv2D_' + str(num + 1), kernel_regularizer=regularizers.l2(reg)))
        if drop:
            model.add(Dropout(rate=dropRate))
        model.add(BatchNormalization(axis=1))
        model.add(Conv2D(filters=filters[num + 1], kernel_size=(3, 3), padding='same', activation='relu',
                         name='Conv2D_' + str(num + 2), kernel_regularizer=regularizers.l2(reg)))

    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # Fully connected network tail:
    model.add(Dense(512, activation='elu', name='FCN_1'))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(Dense(128, activation='elu', name='FCN_2'))
    model.add(Dense(4, activation='softmax', name='FCN_3'))
    model.summary()
    return model


filters = [32, 64, 64, 128, 128]
NNet2 = get_net2(input_shape, drop, dropRate, reg, filters)
NNet2.compile(optimizer=AdamOpt, metrics=['acc'], loss='categorical_crossentropy')
h2 = NNet.fit(x=BaseX_train, y=BaseY_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0,
              validation_data=(BaseX_val, BaseY_val), shuffle=True)

results2 = NNet2.evaluate(X_test, Y_test)
print('test loss, test acc:', results2)

# ----------------------------------------------------------------------------------------


# That's all folks! See you :)
