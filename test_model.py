import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the trained model
model = keras.models.load_model("path_to_saved_model")

# Load a sample image to test the model
image_path = "path_to_test_image"
image = cv2.imread(image_path)
image = cv2.resize(image, (128, 128))
image = np.expand_dims(image, axis=0) / 255.0

# Make predictions
predictions = model.predict(image)
if predictions[0][0] > 0.5:
    print("With mask")
else:
    print("Without mask")
