"""
Test Model
====================

Load a saved model and make predictions on the test data.

Usage
-----
python test_model.py
"""

import cv2 as cv
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils import show_test_batch

# ## Load model

model = tf.keras.models.load_model('../models/spell_detector_model.h5')
model.summary()

# ## Test dataset

TEST_DATASET_PATH = '../data/test/'
IMG_HEIGHT = 100
IMG_WIDTH = 100

# ## Make predictions on the test data

data_generator = ImageDataGenerator(
    rescale=1./255 # Rescale from 0-255 to 0-1
)

test_data_gen = data_generator.flow_from_directory(
    TEST_DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale'
)

test_image_batch, test_label_batch = next(test_data_gen)

predictions = model.predict(test_image_batch)
show_test_batch(test_image_batch, test_label_batch, predictions)