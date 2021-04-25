import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt

dataset = pd.read_csv("fer2013.csv")

X = dataset.iloc[:,1:2].values

y = dataset.iloc[:,0].values

c = np.zeros([35887,2304])
for i in range(len(X)):
    q = X[i][0].split(" ")
    for j in range(2304):
        c[i][j] = q[j]

X = c.reshape(-1,48,48,1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.23,random_state = 0)
x_train = x_train/255.0
x_test = x_test/255.0


from keras import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.optimizers import SGD
from keras.layers import Dropout




classifier = Sequential()

classifier.add(Conv2D(64,(3,3) , input_shape = (48,48,1),activation='relu',))

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Conv2D(64,(3,3) ,activation='relu',))

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

classifier.add(Dense(units = 128,activation='relu'))

classifier.add(Dense(units = 1,activation="sigmoid"))


classifier.compile(optimizer='adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

datagen = ImageDataGenerator(
    featurewise_center=False,  
    samplewise_center=False,  
    featurewise_std_normalization=False,  
    samplewise_std_normalization=False,  
    zca_whitening=False,  
    rotation_range=10,
    zoom_range = 0.1, 
    width_shift_range=0.1, 
    height_shift_range=0.1,
    horizontal_flip=False, 
    vertical_flip=False)

datagen.fit(x_train)

history = classifier.fit(datagen.flow(x_train,y_train), epochs=10, batch_size=32,validation_data=(x_test,y_test))















