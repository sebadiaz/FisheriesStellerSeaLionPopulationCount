from scipy import ndimage
from scipy import misc
from keras.datasets import mnist
from keras.models import Sequential
	
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
np.random.seed(123)  # for reproducibility
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg16
from matplotlib import pyplot as plt


import tensorflow as tf
print(tf.__version__)

image = misc.imread("0.jpg")
print image.shape
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print X_train.shape
xcol=5616
ycol=3744
nbimage=2
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(xcol ,ycol))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img

current=preprocess_image("0.jpg")
print preprocess_image("0.jpg").shape
for i in range(1,nbimage) :
    current=np.concatenate((current,preprocess_image(str(i)+".jpg")), axis=0)
    print str(i)+".jpg"
    print current.shape

plt.imshow(current[0])

train = pd.read_csv('train.csv',usecols=["adult_males", "subadult_males", "adult_females", "juveniles", "pups"],nrows=nbimage)
train = pd.read_csv('train.csv',usecols=["adult_males"],nrows=nbimage)
print train
pool_class = MaxPooling2D
model = Sequential()
model.add(ZeroPadding2D((1,1), input_shape=(xcol,ycol,3)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(pool_class((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(pool_class((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(pool_class((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(pool_class((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(pool_class((2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(current, np.array(train), 
          batch_size=1, nb_epoch=10, verbose=1)

