import tensorflow as tf
from tensorflow import keras

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(augmented_train_ds, epochs=10, validation_data=augmented_test_ds)
