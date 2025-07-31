from queue import Queue
import streamlit as st
import io
import time
import random
from utils.processing_and_inference import process_uploaded_image

st.set_page_config(page_title='Upload', page_icon='images/logo.png', layout='wide', initial_sidebar_state='expanded')
# Initialize session state variable
if 'processing' not in st.session_state:
    st.session_state.processing = False

if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False

if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = False


# session state update functions

def process_button():
    st.session_state.button_clicked = True


## start the page
st.title("Bild-Upload")

with st.container():
    uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["png", "jpg", "jpeg"])
    process_button = st.button('Process Data')
    if process_button and uploaded_file is not None:
        progress_text = "Finding the license plate. Please wait."
        with st.spinner(progress_text, show_time=True):
            detected_license_plate, confidence, inverted_image, uploaded_img = process_uploaded_image(uploaded_file.read())
            # print(inv.shape)
            time.sleep(random.randint(1,5))
        
        if 'prepped_images' not in st.session_state:
            st.session_state.prepped_images = Queue()
        st.session_state.prepped_images.put((detected_license_plate, confidence, inverted_image, uploaded_img))
        
        st.info('You can now upload more pictures, or go the Output page')
    elif process_button and uploaded_file is None:
        st.warning('You have to upload an image')
