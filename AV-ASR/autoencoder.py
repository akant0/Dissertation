from __future__ import print_function
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import keras
import matplotlib.pyplot as plt
import cv2
from scipy.misc import imresize
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Activation
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam
import pickle

# Convolutional autoencoder trained on ultrasound images, used for image feature extraction.

ep=50
# Load images and labels
train_imgs='all-imgs-train-x.npy' # np array of training images
dev_imgs='all-imgs-dev-x.npy' # np array of dev images
train_labels='all-imgs-train-y' # text file with training labels
dev_labels='all-imgs-dev-y' # text file with dev labels

rows, cols = 64, 64 # Image dim.

train_X=np.load(train_imgs) # Load training images
np.random.shuffle(train_X)
scaler = StandardScaler() # Scale for training and dev
train_X=train_X.reshape(train_X.shape[0], rows*cols)
scaler.fit(train_X)
scaler.transform(train_X)
train_X=train_X.reshape(train_X.shape[0], rows, cols, 1)

dev_x=np.load(dev_imgs) # Load dev images
np.random.shuffle(dev_x)
dev_x=dev_x[:10000]

dev_X=dev_x.reshape(dev_x.shape[0], rows*cols)
scaler.transform(dev_X)
dev_X=dev_X.reshape(dev_X.shape[0], rows, cols, 1)
num_train = len(train_X)
height, width, depth = rows, cols, 1

input_shape = (64, 64, 1)
input_img = Input(shape=input_shape)
# adapted from https://blog.keras.io/building-autoencoders-in-keras.html
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(2, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same', name='encoded')(x)

x = Conv2D(2, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer=Adam(),loss='mse')

autoencoder.fit(train_X, train_X,
                 epochs=ep,
                 batch_size=128,
                 shuffle=True,
                 validation_data=(dev_X, dev_X))
print(autoencoder.summary())
encoder_only=Model(input_img, encoded)
encoder_only.save('encoder_only.h5')

