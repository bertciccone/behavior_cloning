import csv

import numpy as np

import cv2
from keras.layers import Cropping2D, Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

# append lines from CSV files from all three sets of driving data
lines = []
nLap1, nLap2, nLap3, nLap4, nLap5 = 0, 0, 0, 0, 0
with open('../mydata/dataA/driving_log.csv') as csvfile:
  # Track 1, 2 laps counter-clockwise + another lap swerving
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
    nLap1 += 1
with open('../mydata/dataB/driving_log.csv') as csvfile:
    # Track 1, 2 laps clockwise + another lap swerving
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
    nLap2 += 1
with open('../mydata/dataC/driving_log.csv') as csvfile:
    # Track 1, 1 lap counter-clockwise, very slow around curves and bridge
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
    nLap3 += 1
with open('../mydata/dataD/driving_log.csv') as csvfile:
    # Track 1, slow only around the dirt curve after bridge
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
    nLap4 += 1
with open('../mydata/dataE/driving_log.csv') as csvfile:
    # Track 1, slow only around the dirt curve after bridge
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
    nLap5 += 1
print('data file', len(lines))

# load images from all three sets of driving data
images = []
measurements = []
for i in range(len(lines)):
  line = lines[i]
  measurement = float(line[3])
  measurements.append(measurement)  # center steering angle
  source_path = line[0]
  filename = source_path.split('/')[-1]
  if (i < nLap1):
    images.append(cv2.imread('../mydata/dataA/IMG/' + filename))
  elif (i < nLap1 + nLap2):
    images.append(cv2.imread('../mydata/dataB/IMG/' + filename))
  elif (i < nLap1 + nLap2 + nLap3):
    images.append(cv2.imread('../mydata/dataC/IMG/' + filename))
  elif (i < nLap1 + nLap2 + nLap3 + nLap4):
    # center image
    images.append(cv2.imread('../mydata/dataD/IMG/' + filename))
    correction = 0.25
    measurements.append(measurement + correction)  # left steering angle
    filename = line[1].split('/')[-1]  # left image
    images.append(cv2.imread('../mydata/dataD/IMG/' + filename))
    measurements.append(measurement - correction)  # right steering angle
    filename = line[2].split('/')[-1]  # right image
    images.append(cv2.imread('../mydata/dataD/IMG/' + filename))
  elif (i < nLap1 + nLap2 + nLap3 + nLap4, + nLap5):
    # center image
    images.append(cv2.imread('../mydata/dataE/IMG/' + filename))
    correction = 0.25
    measurements.append(measurement + correction)  # left steering angle
    filename = line[1].split('/')[-1]  # left image
    images.append(cv2.imread('../mydata/dataE/IMG/' + filename))
    measurements.append(measurement - correction)  # right steering angle
    filename = line[2].split('/')[-1]  # right image
    images.append(cv2.imread('../mydata/dataE/IMG/' + filename))
  assert images[-1].shape == (160, 320, 3), "image shape issue: "

print('images', images[0].shape)

# convert to numpy arrays
X_train = np.array(images)
y_train = np.array(measurements)
print('X_train', X_train.shape, 'y_train', y_train.shape)

epochs = 2
nBatch = 32


def preprocess_image(img):
  img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape(160, 320, 1)
  return img

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image)

# define the model (regression network) - NVIDIA architecture
model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255 - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# build and train the model, mean squared error loss function
model.compile(loss='mse', optimizer='adam')
# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(X_train, y_train, batch_size=nBatch),
                    nb_epoch=epochs,
                    samples_per_epoch=nBatch * len(X_train) / nBatch)
# model.fit(X_train, y_train, validation_split=0.2,
# shuffle=True, batch_size=nBatch, nb_epoch=epochs)

model.save('model19.h5')
