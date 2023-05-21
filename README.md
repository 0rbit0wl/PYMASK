# Mask Detection

This project demonstrates a simple mask detection model using Python, TensorFlow, and Keras. The model can classify whether a person is wearing a mask or not in an image.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- OpenCV

## Installation

1. Clone the repository or download the code.

2. Install the required packages using the following command:
      pip install -r requirements.txt or python3 -m pip install -r requirements.txt

## Dataset

1. Prepare your dataset by organizing it into two folders: `with_mask` and `without_mask`.

2. Update the `dataset_path` variable in `prepare_dataset.py` file to point to the directory containing your dataset.

## Usage

1. Run the `prepare_dataset.py` file to load and split the dataset into training and testing sets.


2. Run the `preprocess_images.py` file to normalize pixel values and apply data augmentation to the dataset.


3. Run the `build_model.py` file to create the mask detection model.


4. Run the `train_model.py` file to compile and train the model using the preprocessed dataset.


5. Once the model is trained, you can use it to make predictions on new images.

- Place the image you want to test in a folder.
- Update the `image_path` variable in `test_model.py` file to point to the test image.
- Run the `test_model.py` file to classify the image.

  ```
  python test_model.py
  ```

The model will output whether the person in the image is wearing a mask or not.

### Real-time Mask Detection

1. Ensure that you have the trained model saved. If not, follow the previous steps to train and save the model.

2. Update the `"path_to_saved_model"` in `video_detect.py` file with the actual path to your trained mask detection model.

3. Make sure you have the Haar cascade file named `haarcascade_frontalface_default.xml` in the same directory as your Python file. If you don't have it, you can download it from the OpenCV GitHub repository: [haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml).

4. Run the `video_detect.py` file to start real-time mask detection using your computer's camera.

A new window will open displaying the camera feed, and the script will draw bounding boxes and labels on faces indicating whether a person is wearing a mask or not. Press 'q' to exit the program.



## Customization

Feel free to customize and experiment with the code to improve the model's performance. You can try different architectures, tune hyperparameters, or use larger datasets for training.

## Acknowledgments

- The code in this project is based on the concepts and techniques from the field of computer vision and deep learning.

- The model's performance can vary based on the quality and diversity of the dataset used for training.

- This project is for educational purposes and serves as a starting point for building a mask detection system.

## License

This project is licensed under the [MIT License](LICENSE).
