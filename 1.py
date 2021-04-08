import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from keras.utils import to_categorical

dataset = pd.read_csv("fer2013.csv")

X = dataset.iloc[:,1:2].values

y = dataset.iloc[:,0].values

y = to_categorical(y)


l = []
for i in range(len(X)):
    l.append(X[i][0].split(" "))  
    
    



from keras import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import MaxPooling2D

classifier = Sequential()

classifier.add(Conv2D(32,(3,3) , input_size = (48,48,3),activation='relu',))

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Conv2D(32,(3,3) ,activation='relu',))

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

classifier.add(Dense(units = 128,activation='relu'))

classifier.add(Dense(units = 1,activation="sigmoid"))

classifier.compile(optimizer='adam',loss = 'binary_crossentropy',metrics = ['accuracy'])


classifier.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=0)


















