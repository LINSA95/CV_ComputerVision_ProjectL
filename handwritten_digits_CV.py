import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical

# STEPS
# Import necessary libraries
# Load and preprocess the MNIST dataset-labelled data set, 60000+10000=train+test data,
# grayscale images of handwritten digits from 0 to 9, size = 28x28 pixel square
# Build the CNN model
# Compile the model
# Train the model
# Evaluate the model
# Make predictions

# Load the MNIST dataset -The MNIST dataset is loaded, split into training and test sets.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
# The images are reshaped to fit the input shape of the CNN and normalized by scaling pixel values to the range [0, 1].
# Labels are one-hot encoded.(x_train.shape[0] = 60000),(x_test.shape[0]=10000)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255 # reshape from (60000,28,28) to (60000,28,28,1),
# 1 represent grey scale image, if it was RGB, then the value will be 3
#(unsigned 8-bit integer) is converted to float32, original pixel value range(0,255) and so converted to range of (0,1)
# by dividing by 255.
# .shape[0] to access 1st element of x.train=tuple(60000,28,28). Here it is 60000
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10) # doing one-hot encoding, integer from 0,9 is changed
# to vector of size 10(3==[0,0,0,1,0,0,0,0,0,0,0])
y_test = to_categorical(y_test, 10)

# Build the CNN model
# Conv2D Layers: Convolutional layers with 32 and 64 filters, using 3x3 kernels and ReLU activation functions.
# MaxPooling2D Layers: Pooling layers to reduce the spatial dimensions.
# Flatten Layer: Converts the 2D matrix to a 1D vector.
# Dense Layers: Fully connected layers, with the last one having 10 neurons (one for each digit) and a
# softmax activation function for classification.
model = Sequential() # initializing sequential, Sequential is a linear stack of layers. You can simply add layers to it one by one.
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))# Adding the First Convolutional Layer
# 32= number of output filters,(3,3)= size of convolutional kernel, 'relu'= applied activation function to the o/p of this layer.
# Rectified linear unit introduces non linearity, (28,28,1) = input size, 28x28 pixel image with 1 channel (grayscale).
model.add(MaxPooling2D(pool_size=(2, 2)))# Adding the First Max Pooling Layer
# Size of the pooling window (2x2). It reduces the spatial dimensions of the feature maps by taking the maximum value over a 2x2 window.
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))# convolutional layers are for grid like structures susch as images
# 64 output layers, 32 used before to extract basic low level features, and then 64 to extract high level features to reduce overfitting
model.add(MaxPooling2D(pool_size=(2, 2)))
#MaxPooling2D layers are added to reduce the spatial dimensions, making the computation more manageable and helping in feature extraction.
model.add(Flatten())# convolutional output will be 3D tensors. SO flatten will make it to 1D tensor
model.add(Dense(128, activation='relu')) # fully connected layer (connected to previous layer) with 128 neurons
# ReLu = gives only positive numbers as output, if negative, it is returned as 0
model.add(Dense(10, activation='softmax'))
# another dense layer with 10 neurons = 10 classes in the data
# sum of probabilities of output from softmax=1, so from each neuron, gets a probability that it belongs to each classes.

# Compile the model-We use the Adam optimizer and categorical cross-entropy loss function.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# adam optimizer - Adaptive Moment Estimation - optimization algorithm used in deeplearning programs
# The method adapts the learning rate of each parameter by estimating the first and second moments of the gradients.
# Adam is efficient, requires little memory, and is well-suited for problems with large datasets and high-dimensional parameter spaces.
# categorical_crossentropy = This is the loss function used for multi-class classification problems where the target variable
# is one-hot encoded.
# It calculates the difference between the actual class (one-hot encoded vector) and the predicted class probabilities.
# The goal is to minimize this difference during training.
# Accuracy is used to evaluate the performance of the model. It is the fraction of correctly predicted instances over
# the total instances.

# Train the model-The model is trained for 10 epochs with a batch size of 200, using the training set and validating on the test set.
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=2)
# An epoch is one complete pass through the entire training dataset. Setting epochs=10 means the training process
# will iterate over the training dataset 10 times.
# Setting batch_size=200 means that the model will take 200 samples at a time, process them, and update the model parameters.
# Using batches instead of processing the entire dataset at once makes training more efficient and allows the model to
# train on datasets that don't fit entirely in memory.
# verbose=2 means that the training process will print one line per epoch. This includes the training and validation loss
# and accuracy for each epoch.
# Evaluate the model-The model’s accuracy is evaluated on the test set.
scores = model.evaluate(x_test, y_test, verbose=0)
# verbose=0 means that the method will run silently without printing any progress or results to the console.
# Other options are verbose=1 (which would print a progress bar) and verbose=2 (which would print one line per batch).
print(scores)
print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
# where scores[0] is the loss and scores[1] is the accuracy of the model on the test data.
# scores[1] is 0.92.
# scores[1] * 100 is 92.0.
# 100 - scores[1] * 100 is 100 - 92.0 = 8.0.

# Make predictions - Predictions are made on the test set, with the first 10 predictions printed.
predictions = model.predict(x_test)
print("First 10 predictions:", predictions[:10])
print("End of Predictions")
