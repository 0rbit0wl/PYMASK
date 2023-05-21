import tensorflow as tf
from tensorflow import keras

# Adjust the paths and labels according to your dataset structure
dataset_path = "path_to_dataset_directory"
train_ds = keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=32,
)
test_ds = keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    batch_size=32,
)
