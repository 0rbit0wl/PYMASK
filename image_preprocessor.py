import tensorflow as tf
from tensorflow import keras

# Normalize pixel values and apply data augmentation
normalization_layer = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)
augmented_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
augmented_test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
