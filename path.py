import os
import streamlit as st

base_dir = os.path.join(os.getcwd(), "Saved Models")
files = os.listdir(base_dir)
st.write("Files in 'Saved Models':", files)