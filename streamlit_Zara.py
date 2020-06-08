import streamlit as st 
from PIL import Image


st.title("Quel est ce vêtement ?")

uploaded_file = st.file_uploader("Choisir une image...", type="jpg, png, jpeg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded', use_column_width=True)
    st.write("")
    st.write("Classification...")
