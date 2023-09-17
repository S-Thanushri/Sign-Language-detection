#IMPORTING BASIC LIBRARIES NEEDED
import csv, os
import numpy as np
import zipfile
import random
import PIL

import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

#Below line specifies that the plots are plotted inside the notebook
%matplotlib inline
from matplotlib import pyplot as plt

#Kaggle dataset is downloaded in a zip file. below commands extract that zip file
zip_path = os.path.join(folder_path, "signlanguage.zip")
zipper = zipfile.ZipFile(zip_path)
zipper.extractall(folder_path)

# Path to csv file for training data
csv_path = os.path.join(folder_path,"sign_mnist_train.csv")

# Reading data and converting into numpy array
raw_data = pd.read_csv(csv_path)
raw_data = raw_data.astype(np.float32)
arr = np.array(raw_data)

# Using same seed so as to get same random shuffle everytime 
random.Random(42).shuffle(arr)

#Spliting the Data into TRAINING AND VALIDATION SETS
split = int(0.8*len(arr))

train_label = arr[:split,0]
train_data = arr[:split,1:]

val_label = arr[split:,0]
val_data = arr[split:,1:]

# Reshaping the data 
train_data = train_data.reshape(-1, 28, 28,1)
val_data = val_data.reshape(-1, 28, 28,1)

#Repeating same procedure to get TEST DATA
test_csv_path = os.path.join(folder_path,"sign_mnist_test.csv")

test_raw_data = pd.read_csv(test_csv_path)
test_raw_data = test_raw_data.astype(np.float32)
test_arr = np.array(test_raw_data)

random.Random(42).shuffle(test_arr)

test_label = test_arr[:,0]
test_data = test_arr[:,1:]

test_data = test_data.reshape(-1, 28, 28,1)

#Plotting a random image to check sanity
plt.imshow(train_data[120])

#Reducing the Data between 0 and 1 so the CNN can perform better
train_data = train_data/(255)
val_data = val_data/(255)
test_data = test_data/(255)

#SANITY CHECK OF DATASETS
print("train data shape:" + str(train_data.shape))
print("train label shape:" + str(train_label.shape))

print("\nValidation data shape:" + str(val_data.shape))
print("Validation label shape:" + str(val_label.shape))

print("\nTest data shape:" + str(test_data.shape))
print("Test label shape:" + str(test_label.shape))

#DEFINING MODEL TO BE USED
model = keras.Sequential([
    layers.Conv2D(32, (5,5), activation = 'relu', input_shape = (28,28,1)),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, (3,3), activation = 'relu', padding = "SAME"),
    layers.MaxPooling2D(2),
    layers.Conv2D(128, (3,3), activation = 'relu', padding = "SAME"),
    layers.Conv2D(128, (3,3), activation = 'relu', padding = "SAME"),
    layers.MaxPooling2D(2),
    layers.Conv2D(256, (3,3), activation = 'relu', padding = "SAME"),
    layers.Conv2D(256, (3,3), activation = 'relu', padding = "SAME"),
    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dropout(0.4),
    layers.Dense(128, activation = 'relu'),
    layers.Dropout(0.4),
    layers.Dense(25, activation = 'softmax')
])

model.summary()

#COMPILING the MODEL
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'rmsProp', metrics = ['acc'])

# FITTING the MODEL onto TRAINING data and Validating on VALIDATION data.
history = model.fit(train_data, train_label, epochs = 30, batch_size = 256,
                        validation_data = (val_data, val_label), verbose = 1)

#EVALUATING the models accuracy on TEST DATA ( which the model has never seen before)
history2 = model.evaluate(test_data,test_label)

#Checking the accuracy and loss on test data
print("Test Accuracy is", history2[1])
print("Test loss is", history2[0])

#PLOTTING THE TRAINING GRAPHS 
acc = history.history['acc']
loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

epoch = range(1, len(acc) + 1)

plt.plot(epoch,acc,"b", label = "Traning Accuracy")
plt.annotate("%0.3f" % acc[-1], xy=(1, acc[-1]),xytext=(2, 0),
            xycoords=('axes fraction', 'data'), textcoords='offset points')
plt.plot(epoch,val_acc,"g", label = "Validation Accuracy")
plt.annotate("%0.3f" % val_acc[-1], xy=(1, val_acc[-1]),xytext=(2, 0),
            xycoords=('axes fraction', 'data'), textcoords='offset points')

plt.legend()
plt.show()

plt.plot(epoch,loss,"r", label = "Traning loss")
plt.annotate("%0.3f" % loss[-1], xy=(1, loss[-1]),xytext=(2, 0),
            xycoords=('axes fraction', 'data'), textcoords='offset points')
plt.plot(epoch,val_loss,"g", label = "Validation loss")
plt.annotate("%0.3f" % val_loss[-1], xy=(1, val_loss[-1]),xytext=(2, 0),
            xycoords=('axes fraction', 'data'), textcoords='offset points')

plt.legend()
plt.show()

#SAVING THE MODEL SO IT CAN BE USED LATER
model.save("C:\\Users\\mukes\\Downloads" + '/SignLanguage.h5')

