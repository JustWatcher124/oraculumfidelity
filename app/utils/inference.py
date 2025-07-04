import pytesseract
import cv2
import numpy
import io
from PIL import Image


def inference_start(image_data: io.BytesIO):
    prepped_image = preprocess_image(image_data)
    
    return prepped_image

def preprocess_image(image_data):

    bytes_to_pil_img = Image.open(io.BytesIO(image_data))
    pil_img_to_np_array = numpy.array(bytes_to_pil_img)

    # Resize the image to 144x600
    resized_img = cv2.resize(pil_img_to_np_array, (600, 144))
    
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    
    # Normalize the pixel values to [0,1]
    normalized_img = gray_img / 255.0
    
    return normalized_img

# todo:
# make the procesing start with the other model
# then use opencv to do the actual stuff
# change code along the way to obfuscate the original author (but thanks for the model lol)


# pil_image = Image.open(io.BytesIO(image_data))
# # pil_image = PIL.Image.open('Image.jpg').convert('RGB')
# open_cv_image = numpy.array(pil_image)
# # Convert RGB to BGR
# image = open_cv_image[:, :, ::-1].copy()
# # image = cv2.imread(image_data)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (3,3), 0)
# thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# # Morph open to remove noise and invert image
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
# invert = 255 - opening

# # Perform text extraction
# data = pytesseract.image_to_string(invert, lang='deu', config='--psm 6')
# data = data.replace('-', '')
# # print(data)