import streamlit as st
from components.ChatEngine import ChatEngine
from components.ContextRetrieval import ContextRetriever

import os
from dotenv import load_dotenv

load_dotenv(override=True)

st.set_page_config(page_title="CDEFG Chatbot", page_icon=":books:")
st.title(":books: CDEFG Workshop Series 22 Jan 2025")
st.header("Sample Chatbot")
st.markdown("Hello World :sunglasses:")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state['messages'] = []

    # Initialized Context Retriever
    st.session_state.retriever = ContextRetriever(
        endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        apikey=os.environ['AZURE_OPENAI_APIKEY'],
        deployment_name=os.environ['AZURE_TEXT_EMBEDDING'],
        model_name=os.environ['AZURE_TEXT_EMBEDDING_MODEL'],
        vs_path="vectorstores/sc1015"
    )

    # Initialized LLM
    st.session_state.llm = ChatEngine(
        endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        apikey=os.environ['AZURE_OPENAI_APIKEY'],
        deployment_name=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
        model_name=os.environ['AZURE_OPENAI_MODEL_NAME'],
        api_version=os.environ['AZURE_OPENAI_API_VERSION'],
        retriever=st.session_state.retriever
    )


# Display chat messages from 
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


# Widget for user input, noted on the equal compare sign :=
if prompt := st.chat_input("Say something"):
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append({"role":"user", "content":prompt})


    # Echo User Response / We will replace this with LLM Engine later
    response = st.session_state.llm.invoke(prompt).content  

    # Display assistant message in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
        st.session_state.messages.append({"role":"assistant", "content":response})



with st.sidebar:
    st.header("Profile")
    gender = st.radio(label="Gender",
                      options = ["Male", "Female"])
    if gender == "Male":
        st.header("Welcome Boys :boy:")
    else:
        st.header("Welcome Girls :girl:")