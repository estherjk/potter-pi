"""
Build Model
====================

Uses TensorFlow to train & build a model to detect spells.

References:
- https://www.tensorflow.org/tutorials/images/classification
- https://www.tensorflow.org/tutorials/load_data/images
- https://www.tensorflow.org/tutorials/keras/classification
- https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
- https://www.tensorflow.org/tutorials/keras/save_and_load

Usage
-----
python build_model.py
"""

from __future__ import print_function

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

from utils import show_batch, show_test_batch

# ## Dataset parameters

DATASET_PATH = '../data/'
TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, 'train')
TEST_DATASET_PATH = os.path.join(DATASET_PATH, 'test')

# TODO: Instead of hard-coding, get them from the datsets themselves. Is it possible w/ the ImageDataGenerator?
NUM_TOTAL_TRAIN = 72
NUM_CLASSES = 3

BATCH_SIZE = 8
NUM_EPOCHS = 5
IMG_HEIGHT = 100
IMG_WIDTH = 100

# ## Data preparation

data_generator = ImageDataGenerator(
    rescale=1./255 # Rescale from 0-255 to 0-1
)

train_data_gen = data_generator.flow_from_directory(
    TRAIN_DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    batch_size=BATCH_SIZE
)

test_data_gen = data_generator.flow_from_directory(
    TEST_DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale'
)

# ## Visualize training images

# Show the next batch
# image_batch, label_batch = next(train_data_gen)
# show_batch(image_batch, label_batch)

# ## Build the model

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=NUM_TOTAL_TRAIN,
    epochs=NUM_EPOCHS
)

model.save('spell_detector_model_new.h5') 

# ## Make predictions

test_image_batch, test_label_batch = next(test_data_gen)
predictions = model.predict(test_image_batch)
show_test_batch(test_image_batch, test_label_batch, predictions)