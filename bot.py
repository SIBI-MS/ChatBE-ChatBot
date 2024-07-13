import os #To access the environment variables
from dotenv import load_dotenv #To load the .env file    
import streamlit as st #To create the web app user interface 
from langchain_core.messages import AIMessage, HumanMessage #Used as schemas for arranging the messages
from langchain_community.utilities import SQLDatabase #To interact with the database     
from langchain_core.prompts import ChatPromptTemplate #Create the prompt for chatbot
from langchain_community.llms import HuggingFaceHub #To create  huggingface model instence
from langchain_community.embeddings import AlephAlphaAsymmetricSemanticEmbedding #Embedding model
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable #for use langsmith to trace with working of code
from langsmith.schemas import Run, Example 
from langsmith.evaluation import evaluate 
from langchain_groq import ChatGroq  
from langsmith.schemas import Run, Example

 
 



load_dotenv() #its for loading the key from the .env file

#Storing the api of huggingface and model name
os.environ["HUGGINGFACE_API_KEY"]=os.getenv("HUGGINGFACE_API_KEY")
model_id="meta-llama/Llama-2-7b-chat-hf"


#To connect with the database
def connect_database(user:str, password:str, host:str, port:str,database:str):
    db_uri=f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)


#create the database chain
def get_sql_chain(db):
    template = """
    
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    
    """
    
    #creating the prompt for make better response
    prompt=ChatPromptTemplate.from_template(template)
    
    #creating an instence for LLM
    llm = HuggingFaceHub(
        huggingfacehub_api_token=os.environ['HUGGINGFACE_API_KEY'],
        repo_id=model_id,
        model_kwargs={"temperature": 0.5, "max_new_tokens": 500}
    )
    
    #To get the schema of database
    def get_schema(_):
        return db.get_table_info()
    
    return(
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )
    
@traceable # Auto-trace this function

#Create function for making the response
def get_response(user_quary:str, db:SQLDatabase,chat_history:list):
    sql_chain=get_sql_chain(db)
    
    #Template for making response as a human by using the uestion, sql query, and sql response
    template = """
    
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response like response by a human.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}
    
    """
    
    #creating the prompt for make better response
    prompt=ChatPromptTemplate.from_template(template)
    
    #creating an instence for LLM
    llm = HuggingFaceHub(
        huggingfacehub_api_token=os.environ['HUGGINGFACE_API_KEY'],
        repo_id=model_id,
        model_kwargs={"temperature": 0.5, "max_new_tokens": 500}
    )
    
    chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
      schema=lambda _: db.get_table_info(),
      response=lambda vars: db.run(vars["query"]),
    )
    | prompt
    | llm
    | StrOutputParser()
  )
    return chain.invoke({
        "question":user_quary,
        "chat_history":chat_history,
    })
    
#chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history=[
        AIMessage(content="Hi, I am ChatBE for your MySQL database. How can I help you?"),
    ]
    

#setting the intial setup of chatbot interface
st.set_page_config(page_title="Chat with Back-End",page_icon=":robot:")
st.title("ChatBE :alien:")

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
            
#storing and handling the user input
user_quary=st.chat_input("Ask you quaery here...")
if user_quary is not None and user_quary!='':
    st.session_state.chat_history.append(HumanMessage(content=user_quary))
    
    with st.chat_message('Human'):
        st.markdown(user_quary)
        
    with st.chat_message('AI'):
        response=get_response(user_quary, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)
    st.session_state.chat_history.append(AIMessage(content=response ))

import numpy as np

# Generate some sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add x0 = 1 to each instance (for the intercept term)
X_b = np.c_[np.ones((100, 1)), X]  # Add x0 = 1 to each instance

# Initialize parameters
theta = np.random.randn(2, 1)  # Random initialization
learning_rate = 0.1
n_iterations = 1000
m = 100  # Number of instances

# Gradient Descent
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients

print("Estimated parameters:", theta)

import pandas as pd

df=pd.read_csv('/home/aiswarya/Documents/student records.csv')

print(df)

feature_names=['OverallGrade', 'Obedient', 'ResearchScore', 'ProjectScore']

training features=df [feature names] print(training features)

outcome name=['Recommend']

outcome_labels df [outcome name]

print(outcome labels)

numeric_feature_names=['ResearchScore', 'ProjectScore'] categorical feature names=['OverallGrade', 'Obedient']

from sklearn.preprocessing import StandardScaler

ss-StandardScaler()

ss.fit(training_features [numeric_feature_names])

training_features [numeric_feature_names]=ss.transform(training_features [numeric_feature names]) print (training features)

training_features=pd.get_dummies (training_features,columns=categorical feature names) print(training features)

