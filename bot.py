import os #To access the environment variables
from dotenv import load_dotenv #To load the .env file
import streamlit as st #To create the web app user interface
from langchain_core.messages import AIMessage, HumanMessage #Used as schemas for arranging the messages
from langchain_community.utilities import SQLDatabase #To interact with the database


load_dotenv()

#To connect with the database
def connect_database(user:str, password:str, host:str, port:str,database:str):
    db_uri=f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)
    
#chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history=[
        AIMessage(content="Hi, I am ChatBE for your MySQL database. How can I help you?"),
    ]
    

#setting the intial setup of chatbot interface
st.set_page_config(page_title="Chat with MySQL",page_icon=":robot:",layout="centered",initial_sidebar_state="expanded")
st.title("ChatMySQL")

#creating the sidebar
with st.sidebar:
    st.subheader('settings')
    st.write("Chat with MySQL")
    
    st.text_input("Host",value="localhost",key="Host")
    st.text_input("Port",value="3366",key='Port')
    st.text_input("User",value="root",key="User")
    st.text_input("Password",value="admin",type="password",key="Password")
    st.text_input("Databese",value="chinnok",key="Database")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db=connect_database(
                st.session_state['User'],
                st.session_state['Password'],
                st.session_state['Host'],  
                st.session_state['Port'],
                st.session_state['Database']
            )
            st.session_state.db=db
            st.success("Connected to database!")


#For showing the chat
for message in st.session_state.chat_history:
    if isinstance(message,AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message,HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
            
            

st.chat_input("Ask you quaery here...")