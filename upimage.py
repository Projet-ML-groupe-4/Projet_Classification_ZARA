#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#UPLOAD IMAGE

from enum import Enum
from io import BytesIO, StringIO
from typing import Union

#import pandas as pd
import streamlit as st
from PIL import Image

st.title('FIND YOUR')
image = Image.open('/Users/ambresaintobert/Downloads/Zara_logo.jpg')
st.image(image, caption='',use_column_width=True)


STYLE = """
<style>
img {
    max-width: 100%;
}
body {
  background-color: white;
}

h1 {
  color: black;
  text-align: center;
}
.reportview-container h1 {
    font-weight: 700;
    font-size: 3.25rem;  
    font-family: Stencil Std;
}
.alert-info{
visibility: hidden;    
}
</style>
"""

FILE_TYPE = ["jpg"]


class FileType(Enum):
    """Used to distinguish between file types"""

    IMAGE = "Image"

def get_file_type(file: Union[BytesIO, StringIO]) -> FileType:
    """The file uploader widget does not provide information on the type of file uploaded so we have
    to guess using rules or ML

    I've implemented rules for now :-)

    Arguments:
        file {Union[BytesIO, StringIO]} -- The file uploaded

    Returns:
        FileType -- A best guess of the file type
    """

    if isinstance(file, BytesIO):
        return FileType.IMAGE
   # content = file.getvalue()

def main():
    """Run this function to display the Streamlit app"""
    st.info(__doc__)
    st.markdown(STYLE, unsafe_allow_html=True)

    file = st.file_uploader("UPLOAD THE IMAGE TO BE ANALYZED (ONLY .JPG):", type=FILE_TYPE)
    show_file = st.empty()
#    if not file:
#        show_file.info("Please upload a file of type: " + ", ".join(FILE_TYPE))
    return

    file_type = get_file_type(file)
    if file_type == FileType.IMAGE:
        show_file.image(file)
        
    file.close()


main()