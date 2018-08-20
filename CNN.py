from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,2"
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import numpy as np
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import sklearn.metrics as metrics

# Convolutional neural network for phone classification of ultrasound images (midsaggital view of child's mouth).
# For dissertation as way to initially analyze ultrasound image ability to classify phones
# (compare accuracy to random chance)
# 41 phone classes can be optionally clustered by: 1. vowels by height/backness, consonants by place of articulation (POA)
# 2. vowels and cons. by POA 3. vowels together, consonants by POA 3. add diphthongs to vowels
# 4. cluster some voiced+nonvoiced consonants by POA

train_imgs='CNN/all-imgs-train-x.npy' # np array of training images
dev_imgs='CNN/all-imgs-dev-x.npy' # np array of dev images
train_labels='CNN/all-imgs-train-y' # text file with training labels
dev_labels='CNN/all-imgs-dev-y' # text file with dev labels
epochs = 5 # number of epochs
LR=.008 # learning rate

rows, cols = 64, 64 # image size

# Cluster phones? Which cluster?
clus=True
cluster1=False
cluster2=True
cluster3=False
cluster4=False
# phone groups
vowels = ['u', 'V', 'O', 'a', 'E', 'I=', 'I', '3', '@', 'i']
Vback = ['u', 'V', 'O']
Vcentral = ['a', 'E', '3', '@']
Vfront = ['I=', 'I', 'i']
cons = ['P', 'b', 'm', 'Z', 'D', 'd', 'T', 'n=', 'r', 't', 'S', 'n', 'z', 's', '5', 'h', 'f', 'v', 'j', 'N', 'g',
        'W', 'k', 'w']
bilabial = ['p', 'b', 'm']
Dental = ['T', 'D']
LD = ['f', 'v']
Alv = ['Z', 'd', 'n=', 'r', 't', 'S', 'n', 'z', 's', '5']
glottal = ['h']
palatal = ['j']
velar = ['N', 'g', 'W', 'k', 'w']
affric = ['dZ', 'tS']
dip = ['eI', 'OI', 'al', 'I@', '@U', 'E@', 'aU', 'ae']
sil = 'sil'

if cluster1==True:
    def phone2clus(phone):
        if phone in vowels or phone in cons:
            if phone in vowels:
                if phone in Vcentral:
                    phone = 'VC'
                if phone in Vfront:
                    phone = 'VF'
                if phone in Vback:
                    phone = 'VB'
            elif phone in cons:
                if phone in bilabial:
                    phone = 'BL'
                if phone in Dental:
                    phone = 'D'
                if phone in LD:
                    phone = 'LD'
                if phone in Alv:
                    phone = 'Alv'
                if phone in velar:
                    phone = 'Vel'
        return phone

if cluster2==True:
    front=Vfront+Dental
    midfront=Alv+Vcentral
    back=velar+palatal+Vback
    misc=glottal+affric+dip+bilabial+LD
    def phone2clus(phone):
        if phone=='sil':
            phone = 'sil'
        if phone in front:
            phone='fr'
        if phone in midfront:
            phone = 'midf'
        if phone in back:
            phone = 'back'
        if phone in misc:
            phone = 'misc'
        return phone
if cluster3==True:
    front=Dental
    midfront=Alv+bilabial+LD
    back=velar+palatal+glottal
    misc=affric
    def phone2clus(phone):
        if phone=='sil':
            phone = 'sil'
        if phone in vowels or phone in dip:
            phone='vowel'
        if phone in midfront:
            phone = 'midf'
        if phone in back:
            phone = 'back'
        if phone in misc:
            phone = 'misc'
        return phone

if cluster4==True:
    def phone2clus(phone):
        if phone in bilabial:
            phone='bil'
        if phone in Dental:
            phone = 'dent'
        if phone =='k' or phone=='g':
            phone= 'kg'
        if phone == 'dZ' or phone == 'tS':
            phone='dZtS'
        if phone =='t' or phone =='d':
            phone = 'td'
        if phone=='s' or phone=='z':
            phone='sz'
        return phone

num2phone={}
phones2Num={}
Num2Phones={}
phones=[]
train_X=np.load(train_imgs) # Load training images
train=[]
train_y=[]
c=0
with open(train_labels, 'r') as f: # Read file with training phone labels
    for i in f:
        i = i.strip()
        phone=i
        if clus==True:
            phone=phone2clus(phone)
        if phone not in phones:
            phones.append(phone)
            phones2Num[phone] = c
            c += 1
        if str(c) not in num2phone:
            num2phone[str(c)]=phone
        train_y.append(phones2Num[phone])
        if phone not in train:
            train.append(phone)

train_Y=np.array(train_y)
scaler = StandardScaler() # Scale/standardize data
train_X=train_X.reshape(train_X.shape[0], rows*cols)
scaler.fit(train_X)
scaler.transform(train_X)
train_X=train_X.reshape(train_X.shape[0], rows, cols)

dev_Y=[]
dev_x=[]

dev_X=np.load(dev_imgs) # Dev images
dev_y=[]
with open(dev_labels, 'r') as f:
    for i in f:
        i = i.strip()
        phone = i
        if clus==True:
            phone=phone2clus(phone)
        dev_Y.append(phones2Num[phone])
num_train=train_X.shape[0]
print(num_train)
num_dev=int(num_train/10)
dev_X=dev_X[:num_dev]
dev_Y=dev_Y[:num_dev]

dev_X=dev_X.reshape(dev_X.shape[0], rows*cols)
scaler.transform(dev_X)
dev_X=dev_X.reshape(dev_X.shape[0], rows, cols)
dev_Y=np.array(dev_Y)

batch_size = 128
num_classes = len(set(train_Y))

train_y= keras.utils.to_categorical(train_Y, num_classes)
dev_Y = keras.utils.to_categorical(dev_Y, num_classes)

train_X = train_X.reshape(train_X.shape[0], rows, cols, 1)
dev_X = dev_X.reshape(dev_X.shape[0], rows, cols, 1)

input_shape = (rows, cols, 1)
train_X = train_X.astype('float32')
dev_X = dev_X.astype('float32')

## Adapted from https://keras.io/getting-started/sequential-model-guide/
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr=LR),
              metrics=['accuracy'])

model.fit(train_X, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(dev_X, dev_Y))

y_pred = model.predict(dev_X)
y_pred_labels = np.argmax(y_pred, axis=1)
confusion_matrix = metrics.confusion_matrix(y_true=np.argmax(dev_Y, axis=1), y_pred=y_pred_labels)
np.set_printoptions(threshold=np.inf)
print(confusion_matrix)
