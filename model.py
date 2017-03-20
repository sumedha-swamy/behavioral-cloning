import csv
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Load the training data.

csv_file = "trainingdata/driving_log.csv"

# Load the CSV file into samples[]
samples = []
with open(csv_file) as f:
    reader = csv.reader(f)
    for row in reader:
        samples.append(row)

# Split the data into training set and test set
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def get_file_name(full_path):
    return ("trainingdata/IMG/" + full_path.split('/')[-1])

# Use a generator so that we dont need to hold all the training data in memory
# at the same time. Instead, we only need as much memory as is needed to hold
# one batch of training data
def generator(samples, batch_size=32):
    num_samples = (int(len(samples)/batch_size))*batch_size
    correction = 0.2 # 0.2 seems to work well as a correction factor. Tunable in future
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(get_file_name(batch_sample[0]))
                center_angle = float(batch_sample[3])
            
                left_image = cv2.imread(get_file_name(batch_sample[1]))
                left_angle = center_angle + correction
            
                right_image = cv2.imread(get_file_name(batch_sample[2]))
                right_angle = center_angle - correction
              
                images.extend([center_image, left_image, right_image])
                angles.extend([center_angle, left_angle, right_angle])
                

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
            

# Build a Model using Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

model = Sequential()

# Use a lambda layer to normalize the image - useful to do this so that we can
# leverage the GPU to normalize images in parallel
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# We dont need the top part of the image which is the sky. We also dont need
# the bottom part which is a view of the dashboard. Crop those parts
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))

# Convolutional Layers
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.1))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(0.1))
model.add(Convolution2D(64, 3, 3, activation="relu"))

# Fully connected layers
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.1))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

# Fit the model
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
history_object = model.fit_generator(train_generator, samples_per_epoch= (int(len(train_samples)/32))*32,
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples), 
                                     nb_epoch=7)

model.save('model.h5')
print("Model saved")


# Print Training progress
print(history_object.history.keys())

# Plot the training and validation loss for each epoch
import matplotlib.pyplot as plt
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()