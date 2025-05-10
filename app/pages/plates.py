import streamlit as st
st.set_page_config(page_title='Kennzeichen Kreation', page_icon='images/logo.png', layout='wide', initial_sidebar_state='expanded')


from utils.lplates import get_list_license_plates


