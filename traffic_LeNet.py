import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import PIL
from tensorflow.keras import layers
import pandas as pd
import seaborn as sns
import pickle
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

# Load the dataset of file format .p
train_data = pickle.load(open('traffic-signs-data/train.p', 'rb'))
test_data = pickle.load(open('traffic-signs-data/test.p', 'rb')) 
valid_data = pickle.load(open('traffic-signs-data/valid.p', 'rb'))

# Extract the features and labels
X_train, y_train = train_data['features'], train_data['labels']
x_valid, y_valid = valid_data['features'], valid_data['labels']
X_test, y_test = test_data['features'], test_data['labels']


# Check the shape of the data
print(X_train.shape)
print(X_test.shape)
print(x_valid.shape)

# visualize an image
i = 2100

plt.imshow(X_train[i])  
plt.title(f"Label: {y_train[i]}")  
plt.axis("off") 
plt.show()

# Shuffle the data to avoid bias in the model. This is done to avoid the model from learning the order of the data and not the features.
X_train, y_train = shuffle(X_train, y_train)

# Convert the images to grayscale
X_train_gray = np.sum(X_train/3, axis=3, keepdims=True)
X_test_gray = np.sum(X_test/3, axis=3, keepdims=True)
x_valid_gray = np.sum(x_valid/3, axis=3, keepdims=True)

#normalize the data
X_train_gray_norm = (X_train_gray - 128)/128
X_test_gray_norm = (X_test_gray - 128)/128
x_valid_gray_norm = (x_valid_gray - 128)/128

# Check the difference between the original and grayscale images and normalized images
i = 610
plt.imshow(X_train[i])
plt.title(f"Label: {y_train[i]}")
plt.axis('off')
plt.show()

plt.imshow(X_train_gray[i].squeeze(), cmap='gray')
plt.title(f"Label: {y_train[i]}")
plt.axis('off')
plt.show()

plt.imshow(X_train_gray_norm[i].squeeze(), cmap='gray')
plt.title(f"Label: {y_train[i]}")
plt.axis('off')
plt.show()

# Build the model
from tensorflow.keras import datasets, layers, models

LeNet = models.Sequential()
LeNet.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 1))) #here 6 is the number of filters and (5,5) is the filter size.
LeNet.add(layers.AveragePooling2D(pool_size=(2, 2), strides=2))  
LeNet.add(layers.Conv2D(16, (5, 5), activation='relu')) #here 16 is the number of filters and (5,5) is the filter size.
LeNet.add(layers.AveragePooling2D(pool_size=(2, 2), strides=2))  
LeNet.add(layers.Flatten())
LeNet.add(layers.Dense(120, activation='relu'))
LeNet.add(layers.Dense(84, activation='relu'))
LeNet.add(layers.Dense(43, activation='softmax')) #43 is the number of classes

LeNet.summary()

#compile the model
LeNet.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) #sparse categorical is used when the labels are integers

#train the model
history = LeNet.fit(X_train_gray_norm, y_train, batch_size=500, epochs=50, verbose =1,  validation_data=(x_valid_gray_norm, y_valid)) #verbose is used to display the output of the model


# Evaluate the model
history.history.keys()

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training Accuracy') 
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy') 
plt.plot(epochs, loss, 'ro', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Accuracy and Loss')
plt.legend()
plt.show()

# Test the model
predicted_classes = np.argmax(LeNet.predict(X_test_gray_norm), axis=-1)
y_true = y_test


#create a confusion matrix
cm = confusion_matrix(y_true, predicted_classes)
plt.figure(figsize = (25, 25))
sns.heatmap(cm, annot=True)


L = 7
W = 7
fig, axes = plt.subplots(L, W, figsize = (12, 12))
axes = axes.ravel() 
for i in np.arange(0, L * W):  
    axes[i].imshow(X_test[i])
    axes[i].set_title(f"Prediction={predicted_classes[i]}\n True={y_true[i]}")
    axes[i].axis('off')
plt.subplots_adjust(wspace=1)
plt.show()


