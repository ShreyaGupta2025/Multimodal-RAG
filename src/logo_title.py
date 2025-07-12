
import streamlit as st

def logo_title(title):
    st.title(title)
    logo_path = "images/bits_logo.png"  # Replace with the path to your logo file
    st.sidebar.image(logo_path)
    st.sidebar.markdown("<style>h1 { margin-top: 0; }</style>", unsafe_allow_html=True)
    st.sidebar.markdown("<h1 style='text-align: center;'>GenAI Application</h1>", unsafe_allow_html=True)

  

