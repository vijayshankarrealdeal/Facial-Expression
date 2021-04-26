import pandas as pd
import numpy as np

dataset = pd.read_csv('fer2013.csv')

y_train = dataset['emotion'].values
x_train = dataset['pixels'].values
usage   = dataset['Usage']
import seaborn as sns

sns.countplot(y_train)

X_train = []
for i in range(len(x_train)):
  X_train.append(x_train[i].split(' '))

X_train = np.array(X_train,dtype = 'float32')

X_train = X_train/255.0
X_train =X_train.reshape(-1,48,48,1)

import matplotlib.pyplot as plt

plt.imshow(X_train[207][:,:,0])

from keras.utils import to_categorical
y_train = to_categorical(y_train)

from sklearn.model_selection import  train_test_split
X_train, X_val, Y_train,Y_val = train_test_split(X_train, y_train, test_size = 0.23, random_state=0)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.layers import Dropout

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau 


model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3) ,padding="Same",activation="relu",input_shape=(48,48,1)))
model.add(Conv2D(filters=32,kernel_size=(3,3) ,padding="Same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(7, activation = "softmax"))
optimizer = Adam()

model.compile(optimizer, loss = "categorical_crossentropy",metrics=["accuracy"])


learning_rate_reduction = ReduceLROnPlateau(monitor="val_loss",
                                            patience=1, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)
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

datagen.fit(X_train)

epochs = 425
batch_size = 86

history = model.fit(
    datagen.flow(X_train,Y_train),
    batch_size=batch_size,
    validation_data = (X_val,Y_val),
    epochs = epochs,
    verbose = 1,
    shuffle = True,
    callbacks=[learning_rate_reduction],
    steps_per_epoch=X_train.shape[0]//batch_size,
)
x = range(epochs)

loss_train = history.history['loss']
loss_val = history.history['val_loss']
plt.plot(x, loss_train, 'g', label='Training loss')
plt.plot(x, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

plt.plot(x, accuracy, 'r', label='Training accuracy')
plt.plot(x, val_accuracy, 'y', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()