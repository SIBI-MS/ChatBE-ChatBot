import os 
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

#setting the intial setup of chatbot interface
st.set_page_config(page_title="Chat with MySQL",page_icon=":robot:",layout="centered",initial_sidebar_state="expanded")
st.title("ChatMySQL")

#creating the sidebar
with st.sidebar:
    st.subheader('settings')
    st.write("Chat with MySQL")
    
    st.text_input("Host",value="localhost")
    st.text_input("Port",value="3366")
    st.text_input("User",value="root")
    st.text_input("Password",value="admin")