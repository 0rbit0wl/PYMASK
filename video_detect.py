import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the trained model
model = keras.models.load_model("path_to_saved_model")

# Define the labels for mask and no mask
LABELS = ["Without Mask", "With Mask"]

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Define the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Loop over frames from the video stream
while True:
    # Read the frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Preprocess the face region
        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, (128, 128))
        face = np.expand_dims(face, axis=0) / 255.0

        # Make predictions on the face
        predictions = model.predict(face)
        label = LABELS[int(predictions[0][0])]

        # Determine the label and color for bounding box
        if label == "With Mask":
            color = (0, 255, 0)  # Green color for with mask
        else:
            color = (0, 0, 255)  # Red color for without mask

        # Draw the bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the resulting frame
    cv2.imshow("Mask Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
