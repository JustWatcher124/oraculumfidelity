# import pytesseract
import cv2
import numpy as np
import io
from PIL import Image
from ultralytics import YOLO
from model_def import model_definition
import itertools

MODEL_FILENAME = 'data/best_license_plate_model.pt'

def process_uploaded_image(image_data: io.BytesIO, model):

    detected_license_plate, confidence = detect_lp(image_data, model=model)
    detected_license_plate = Image.open(io.BytesIO(image_data))
    print(np.array(detected_license_plate).shape)
    if not check_image_size(np.array(detected_license_plate)):  # image is too small - upscales
        print('is upscaling')
        print(np.array(detected_license_plate).shape)
        upscaled_image = upscale_image(detected_license_plate)
        resized_image = resize_image(upscaled_image)
    else:  # down-size image
        upscaled_image = None
        resized_image = resize_image(detected_license_plate)
    # resized_image = resize_image(np.array(detected_license_plate))
    inverted_image = preprocess_image(resized_image)
    
    return detected_license_plate, 0, inverted_image


def inference(processed_img):
    
    correct_dimensioned_img = check_dimensions(processed_img)
    if correct_dimensioned_img is None:
        return None
    WEIGHTS='data/model_final.weights.h5'
    model = model_definition()
    model.load_weights(WEIGHTS)
    
    prediction = model.predict(correct_dimensioned_img)
    decoded = decode_label(prediction)
    
    return decoded


#https://keras.io/examples/image_ocr/
#https://github.com/qjadud1994/CRNN-Keras
def decode_label(out):
    """
    Takes the predicted ouput matrix from the Model and returns the output text for the image
    """
    # out : (1, 48, 37)
    out_best = list(np.argmax(out[0,2:], axis=1))

    out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value

    outstr=words_from_labels(out_best)
    return outstr


def words_from_labels(labels):
    """
    converts the list of encoded integer labels to word strings like eg. [12,10,29] returns CAT 
    """
    letters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZÖÜ'
    txt=[]
    for ele in labels:
        if ele == len(letters): # CTC blank space
            txt.append("")
        else:
            #print(letters[ele])
            txt.append(letters[ele])
    return "".join(txt)


def check_dimensions(image_data):
    if image_data.shape != (1, 90, 160, 1):
        if image_data.shape == (1, 90, 160):
            image_data = image_data[..., np.newaxis]
        elif image_data.shape == (90, 160, 1):
            image_data = image_data[np.newaxis, ...]
        elif image_data.shape == (90, 160):  # when coming from func process_uploaded_image
            image_data = image_data[np.newaxis, ..., np.newaxis]
        else:  # give up, some other shape was given
            image_data = None
    return image_data
    

def preprocess_image(PIL_img):

    #bytes_to_pil_img = Image.open(io.BytesIO(image_data))
    pil_img_to_np_array = np.array(PIL_img)

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(pil_img_to_np_array, cv2.COLOR_BGR2GRAY)

    # Normalize the pixel values to [0,1]
    # blurred_img = cv2.GaussianBlur(gray_img, (3,3), 0)
    applied_threshold = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
   
    # Morph open to remove noise and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    opening = cv2.morphologyEx(applied_threshold, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening  #  black text and white background
    return invert


def detect_lp(image_data, model=None):
    bytes_to_pil_img = Image.open(io.BytesIO(image_data))
    if model is None:
        model = YOLO(MODEL_FILENAME)  # loads a trained model
    # Perform prediction on the test image using the model
    results = model.predict(bytes_to_pil_img, device='cpu', verbose=False)
    # Load the image using OpenCV
    #image = cv2.imread(np.array(bytes_to_pil_img))
    image = np.array(bytes_to_pil_img)

    if image is None:
        print(f"Error: Unable to load image")
        return -1

    # Convert the image from BGR (OpenCV default) to RGB (matplotlib default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cutouts = []

    # Check if results contain any detections
    if not results:
        print("No detections found.")
        return -2

    # Extract the bounding boxes and labels from the results
    for result in results:
        # Ensure `result.boxes` exists and is iterable
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                # Get the coordinates of the bounding box
                if hasattr(box, 'xyxy') and hasattr(box, 'conf'):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]

                    # Draw the bounding box on the image
                    #cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw the confidence score near the bounding box
                    cv2.putText(image, f'{confidence*100:.2f}%', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                    # Get the cutout of the bounding box
                    cutout = image[y1:y2, x1:x2]

                    # Save the cutout as a separate image
                    cutouts.append((Image.fromarray(cutout), confidence*100))
    if len(cutouts) == 0:  # no things detected for some reason
        # fail forward and give the text-rec a chance
        likeliest_lp = (image, 0)
    else:   # found a plate (hopefully normal case)
        likeliest_lp = max(cutouts, key=lambda x: x[1])
    return likeliest_lp



def upscale_image(img):
    MODEL_PATH = "data/EDSR_x4.pb" # Your model path, download it above
    MODEL_NAME = "edsr"              # Can be "espcn", lapsrn", "fsrcnn" or "edsr"
    SCALE = 4                        # The model scale
    
    if not isinstance(img, type(np.array([1]))):
        image = np.array(img)
    else:
        image = img
    #image = cv2.imread(IMAGE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sr_model = cv2.dnn_superres.DnnSuperResImpl_create()
    sr_model.readModel(MODEL_PATH)
    sr_model.setModel(MODEL_NAME,SCALE)
    
    upsampled_image = sr_model.upsample(image)
    return upsampled_image

def check_image_size(image_array):
    # Get the shape of the numpy array
    height, width, _ = image_array.shape
    
    # Compare the dimensions to 160x90
    return (width >= 155 and width <= 165) and (height >= 86 and height <= 94)

def resize_image(image_array):
    # Convert the image array to a format compatible with OpenCV
    image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    # Define the desired output size
    target_size = (160, 90)
    # Use OpenCV's resize function to get the resized image
    resized_image = cv2.resize(image, target_size)
    
    return resized_image
