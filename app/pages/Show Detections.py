import streamlit as st
# from queue import Queue
from utils.processing_and_inference import inference, inference_faking
st.set_page_config(page_title='Output', page_icon='images/logo.png', layout='wide', initial_sidebar_state='expanded')
st.title("Detections And Output")

if 'prepped_images' not in st.session_state:
    st.info("No images uploaded. Please go to the Upload page first.")
    st.stop()  # Halts execution of this page


if 'inferenced' not in st.session_state:
    st.session_state.inferenced = []


if 'inferenced_f' not in st.session_state:
    st.session_state.inferenced_f = []
    
    
with st.container():
    if not st.session_state.prepped_images.empty():
        st.write('There are non-inferenced, but preprocessed images in storage.')
        inference_lps = st.button('Read the Licence Plates')
        if inference_lps:
            with st.spinner('Reading the License Plates...'):
                st.info('Don\'t leave this page until it updates')
                while not st.session_state.prepped_images.empty():
                    detected_lp, confidence, inverse, uploaded_img =  st.session_state.prepped_images.get()
                    detected_text = inference_faking(detected_lp)
                    st.session_state.inferenced_f.append((detected_lp, uploaded_img ,detected_text))
                    detected_text = inference(inverse)
                    st.session_state.inferenced.append((detected_lp, uploaded_img ,detected_text))
                inference_lps = None
    st.divider()
    # if not st.session_state.inferenced:
    #     st.info('Nothing to Show yet')
    #     st.stop()
    # for idx, (lp, uploaded_img, text) in enumerate(st.session_state.inferenced):
    #     with st.container(border=True):
    #         # st.image(inverse)
    #         st.write('Uploaded Image:')
    #         st.image(uploaded_img)
    #         st.write('Detected Licenseplate:')
    #         st.image(lp)
    #         st.write(f'Read License Plate: {text}')
    if not st.session_state.inferenced_f:
        st.info('Nothing to Show yet')
        st.stop()
    st.info('Here are all images that you uploaded and have had processed through LASY')
    for idx, (lp, uploaded_img, text) in enumerate(st.session_state.inferenced_f):
        with st.container(border=True):
            # st.image(inverse)
            st.write('The image you uploaded:')
            st.image(uploaded_img)
            #st.write('Detected Licenseplate:')
            #st.image(lp)
            st.write(f'LASY has detected following license plate:')
            st.markdown(f"<h1 style='font-size: 2em;'>{text}</h1>", unsafe_allow_html=True)
    # st.session_state.prepped_images.append((detected_license_plate, confidence, inv))
