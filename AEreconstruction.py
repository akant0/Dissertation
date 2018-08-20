from __future__ import print_function
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
print(matplotlib.get_backend())
import cv2
import os
from scipy.misc import imresize
from sklearn.preprocessing import StandardScaler
from keras.datasets import mnist
from keras.models import Model
import pickle
from keras.models import load_model

loc='all/'
name='all'

scalerfile = loc+'scaler-'+name+'.sav'
scaler = pickle.load(open(scalerfile, 'rb'))
autoencoder=load_model(loc+'autoencoder-'+name+'.h5')
dev_x=[]

# Pick which image to reconstruct
img = '/media/alexandra/Seagate Game Drive/US-IMAGES-COMPLETE/UXTD/01M/01M-001B-1.png'
img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
img=cv2.resize(img, (64,64))

dev_x.append(img)
dev_X = np.array(dev_x).astype('float32')/255.0
dev_X=dev_X.reshape(dev_X.shape[0], 64*64)
scaler.transform(dev_X)

print(dev_X.shape)
dev_X=dev_X.reshape(dev_X.shape[0], 64, 64, 1)

print(dev_X.shape)
decoded_imgs = autoencoder.predict(dev_X)

# save the original image and reconstruction
i=0
plt.figure()
fig,ax2 = plt.subplots(1)
plt.imshow(dev_X[i].reshape(64,64))
plt.gray()
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
plt.savefig('orig.svg', format="svg")

plt.figure()
fig,ax2 = plt.subplots(1)
plt.imshow(decoded_imgs[i].reshape(64, 64))
plt.gray()
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
plt.savefig('reconstructed.svg', format="svg")
