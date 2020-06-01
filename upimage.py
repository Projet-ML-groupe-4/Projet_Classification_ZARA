#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:08:50 2020

@author: ambresaintobert
"""


#UPLOAD IMAGE

from enum import Enum
from io import BytesIO, StringIO
from typing import Union

#import pandas as pd
import streamlit as st

STYLE = """
<style>
img {
    max-width: 100%;
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

    file = st.file_uploader("Chargez l'image de votre vêtement", type=FILE_TYPE)
    show_file = st.empty()
    if not file:
        show_file.info("Please upload a file of type: " + ", ".join(FILE_TYPE))
        return

    file_type = get_file_type(file)
    if file_type == FileType.IMAGE:
        show_file.image(file)
        
    file.close()


main()