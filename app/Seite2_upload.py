
import streamlit as st
from PIL import Image
import io

def show():
    st.title("Bild-Upload")
    
    uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))
        st.image(image, caption="Hochgeladenes Bild", use_column_width=True)
        st.success("Bild erfolgreich verarbeitet.")