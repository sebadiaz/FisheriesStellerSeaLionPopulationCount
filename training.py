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
model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(xcol,ycol,3)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(current, np.array(train), 
          batch_size=1, nb_epoch=10, verbose=1)
