# Import necessary libraries
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from PIL import Image
import os
import re

class LicensePlateRecognizer:
    def __init__(self):
        self.model = self.new_model()

    def create_model(self):
        # Create a convolutional neural network model
        model = Sequential()
        model.add(Conv2D(32, (7, 7), activation='relu', input_shape=(144, 600, 4)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    def new_model(self):
        model = Sequential()
        # convolutional layers
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(90,160, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        # Flatten
        model.add(Flatten())

        # Dense layers
        model.add(Dense(128, activation='relu'))
        model.add(Dense(36, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=5, batch_size=32)

    def predict(self, image_path):
        # Load the image
        img = Image.open(image_path)
        img = np.array(img.convert('RGBA'))
        img = np.expand_dims(img, axis=-1)  # Add channel dimension

        # Resize the image to match the model's input shape
        img = np.resize(img, (28, 28))

        # Normalize the pixel values
        img = img / 255.0

        # Make predictions
        predictions = self.model.predict(np.array([img]))
        return predictions.argmax()

def extract_license_plate_from(filename):
    # Remove file extension and split at spaces
    plate_chars = filename.split('.')[0]
    
    short_code, letters = plate_chars.split('-')[:2]
    letters = letters.split(' ')[0]
    numbers = plate_chars.split(' ')[-1]
    return short_code, letters, numbers


def get_license_plate_character_set(filename):
    step_one = plate_chars = filename.split('.')[0]
    step_two = step_one.replace('-', '')
    step_three = step_two.replace(' ', '')
    return step_three

def get_image_dataset(path='outputs/'):
    """
    Load a dataset of images from a directory.

    Args:
        path (str): Path to the directory containing the images.

    Returns:
        X (np.ndarray): A numpy array of shape (num_images, height, width, channels) containing the loaded images.
        y (list): A list of corresponding labels for each image.
    """
    
    # Initialize empty lists to store the images and their corresponding labels
    X = []
    y = []

    # Walk through all files in the specified directory
    for filename in os.listdir(path):
        # Check if it's an image file (assuming .png extension)
        if filename.lower().endswith('.png'):
            # Construct the full path to the image file
            image_path = os.path.join(path, filename)
            
            # Load the image and add a channel dimension (assuming grayscale images)
            img = np.array(Image.open(image_path).convert('RGBA'))
            img = np.expand_dims(img, axis=-1)

            # Store the image and its corresponding label in the lists
            X.append(img)
            # y.append(filename[:3])  # Assuming the first three characters of each filename represent the license plate's letters
    labels = {}
    for i, label in enumerate(sorted(set(y))):
        labels[label] = i
    
    y_encoded = np.zeros((len(X), len(labels)))
    for i, label in enumerate(y):
        y_encoded[i, labels[label]] = 1
    
    return X, y_encoded



# Example usage:
filename = 'B-AA 123.png'
plate_chars = get_license_plate_character_set(filename)
print(plate_chars)  # Output: ['B', 'AA', '123']


# Function to create multi-hot encoded label
def create_multi_hot_label(plate):
    label = [0] * len(char_to_index)
    
    # Mark the presence of each character in the plate
    for char in plate:
        index = char_to_index[char]
        label[index] = 1
    
    return label



X_train, y_train = get_image_dataset()
# Example usage:
#recognizer = LicensePlateRecognizer()


#X_train = []  # Replace with your training images
#y_train = []  # Replace with your corresponding labels (e.g., letter/number values)
# recognizer.train_model(X_train, y_train)

# image_path = 'A-YZ 4281.png'
# predicted_label = recognizer.predict(image_path)
# print(predicted_label)