# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from PIL import Image
import os
# import re
import pandas as pd
from sklearn.model_selection import train_test_split
from inference import process_uploaded_image
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
# class LicensePlateRecognizer:
#     def __init__(self):
#         self.model = self.new_model()

#     def create_model(self):
#         # Create a convolutional neural network model
#         model = Sequential()
#         model.add(Conv2D(32, (7, 7), activation='relu', input_shape=(144, 600, 4)))
#         model.add(MaxPooling2D((2, 2)))
#         model.add(Conv2D(64, (3, 3), activation='relu'))
#         model.add(MaxPooling2D((2, 2)))
#         model.add(Conv2D(128, (3, 3), activation='relu'))
#         model.add(MaxPooling2D((2, 2)))
#         model.add(Flatten())
#         model.add(Dense(64, activation='relu'))
#         model.add(Dense(10, activation='softmax'))

#         # Compile the model
#         model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#         return model
#     def new_model(self):
#         model = Sequential()
#         # convolutional layers
#         model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(90,160, 3)))
#         model.add(MaxPooling2D((2, 2)))
#         model.add(Conv2D(64, (3, 3), activation='relu'))
#         model.add(MaxPooling2D((2, 2)))
#         model.add(Conv2D(128, (3, 3), activation='relu'))
#         model.add(MaxPooling2D((2, 2)))

#         # Flatten
#         model.add(Flatten())

#         # Dense layers
#         model.add(Dense(128, activation='relu'))
#         model.add(Dense(36, activation='softmax'))

#         # Compile the model
#         model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#     def train_model(self, X_train, y_train):
#         self.model.fit(X_train, y_train, epochs=5, batch_size=32)

#     def predict(self, image_path):
#         # Load the image
#         img = Image.open(image_path)
#         img = np.array(img.convert('RGBA'))
#         img = np.expand_dims(img, axis=-1)  # Add channel dimension

#         # Resize the image to match the model's input shape
#         img = np.resize(img, (28, 28))

#         # Normalize the pixel values
#         img = img / 255.0

#         # Make predictions
#         predictions = self.model.predict(np.array([img]))
#         return predictions.argmax()
    
    
def extract_license_plate_from(filename):
    # Remove file extension and split at spaces
    plate_chars = filename.split('.')[0]
    
    short_code, letters = plate_chars.split('-')[:2]
    letters = letters.split(' ')[0]
    numbers = plate_chars.split(' ')[-1]
    return ''.join([short_code, letters, numbers])



def merge_datasets():
    path1 = '../dataset/text_rec/images/'
    path2 = './outputs/'
    csv_file = '../dataset/text_rec/lpr.csv'
    # Load CSV file into dataframe
    df = pd.read_csv(csv_file)
    label_lookup = df[['images', 'labels']].set_index('images').to_dict()['labels']
    image_paths = []
    labels = []
    
    # split names and labels
    for file in os.listdir(path1):
        if file.endswith(".jpg"):
            # filename = file.split('/')[-1]
            image_paths.append(os.path.join(path1, file))
            labels.append(label_lookup[file])

    # Add images from testing dataset to lists
    for file in os.listdir(path2):
        if file.endswith(".png"):
            # filename = file.split('/')[-1]
            lp_text = extract_license_plate_from(file)
            image_paths.append(os.path.join(path2, file))
            labels.append(lp_text)

   
    # Split the data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


X_train_paths, X_test_paths, y_train, y_test = merge_datasets()


def process_images(list_of_paths, model, label):
    img_data = []
    for img_path in list_of_paths:
        with io.open(img_path, 'rb') as image_file:
            image_data = image_file.read()
        # print(img_path)
        detected_license_plate, confidence, inverted_image = process_uploaded_image(image_data, model)
        
        img_data.append(inverted_image)
    return img_data, label


# def process_data(path):
    
#     return inverted_image

# def chunks(lst, n):
#     """Yield successive n-sized chunks from lst."""
#     for i in range(0, len(lst), n):
#         yield lst[i:i + n]

# # X_test = process_images(X_test_paths)

# # process_name = f'{X_train_paths=}'.split('=')[0]
from ultralytics import YOLO
MODEL_FILENAME = 'data/best_license_plate_model.pt'
model = YOLO(MODEL_FILENAME)
def do_some_threading(X, y, name):
    
    # Create a thread pool and run the function for each path in the subgroups
    with ThreadPoolExecutor(max_workers=14) as executor:
        results = []  # Initialize an empty list to store the results
        futures = []
        n = 50
        
        for (subgroup, label) in [(X[i:i + n], y[i:i + n]) for i in range(0, len(X), n)]:
            # print(f"Processing subgroup: {subgroup_dir}")
            futures.append(executor.submit(process_images, subgroup, model, label))
            # break
             # = future
        
        # As each task completes, append its result to the list
        for _ in as_completed(futures):
            try:
                result = next(as_completed(futures)).result()
                results.append(result)  # Append the result to the list
                print(len(results))
            except Exception as e:
                print(f"Error processing: {e}")
    
    
    with open(name, "wb") as fp:
        pickle.dump(results, fp)

do_some_threading(X_train_paths, y_train, 'training_data.pkl')
do_some_threading(X_test_paths, y_test, 'testing_data.pkl')

def flatten_concatenation(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list

with open('training_data.pkl', "rb") as fp:
    X_train_mat = pickle.load(fp)
    # df_t = pd.DataFrame(X_train_mat)
X_train_ls = [tup[0]  for tup in X_train_mat]
y_train_ls = [tup[1]  for tup in X_train_mat]
X_train = flatten_concatenation(X_train_ls)
y_train = flatten_concatenation(y_train_ls)


with open('testing_data.pkl', "rb") as fp:
    X_test_mat = pickle.load(fp)
X_test_ls = [tup[0]  for tup in X_test_mat]
y_test_ls = [tup[1]  for tup in X_test_mat]
X_test = flatten_concatenation(X_test_ls)
y_test = flatten_concatenation(y_test_ls)


# X_test = flatten_concatenation(X_test_mat)
    

