import streamlit as st
from pathlib import Path

PAGES = {
    "Homepage": "Seite1_home",
    "Bild hochladen": "Seite2_upload",
    "Output": "Seite3_output",
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Gehe zu", list(PAGES.keys()))

module = __import__(PAGES[selection])
module.show()
