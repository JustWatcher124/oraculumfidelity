
import streamlit as st
import io
import time
import random
from utils.inference import inference_start

# Initialize session state variable
if 'processing' not in st.session_state:
    st.session_state.processing = False

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False

if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = False


# session state update functions
def uploaded_or_changed_image():
    st.session_state.image_uploaded = True

def process_button():
    st.session_state.button_clicked = True


## start the page
st.title("Bild-Upload")

with st.container():
    uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["png", "jpg", "jpeg"], on_change=uploaded_or_changed_image)
    st.button('Process Data', on_click=process_button)
    if st.session_state.button_clicked and st.session_state.image_uploaded:
        st.session_state.processing = True
        result = inference_start(uploaded_file.read())
        # If user stays on this page, the result will be displayed below the button
        if st.session_state.processing:
            st.image(result, use_container_width=True)
    elif st.session_state.button_clicked and not st.session_state.image_uploaded:
        st.warning('You have to upload an image')
        st.session_state.button_clicked = False


        