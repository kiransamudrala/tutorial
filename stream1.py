import streamlit as st
import numpy as np
import pandas as pd

st.title('The Gungan Planet, Naboo!')
user_input = ' '

user_input = st.text_input("What is youssa name?", ' ')


if user_input==' ':
    st.write('')
else:
    st.write('Messa welcomes youssa to our great planet Naboo, ',user_input, '!')

from PIL import Image
image = Image.open('jjb.jpg')
st.image(image, caption='Messa Jar Jar Binks!',use_column_width=True)
