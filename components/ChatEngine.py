from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# from ContextRetrieval import ContextRetriever


class ChatEngine:
    def __init__(self, endpoint, apikey, deployment_name, model_name, api_version, retriever=None, temperature=0):
        self.llm = AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=apikey,
            deployment_name=deployment_name,
            model_name=model_name,
            api_version=api_version,
            temperature=0
        
        )

        persona = "You are a teaching assistant at for the course SC1015 at NTU."
        task ="your task is to answer student query about the data science and ai course."
        context = "the context will be provided based on the course information and FAQ along with the user query"
        condition = "If user ask any query beyond data science and ai, tell the user you are not an expert of the topic the user is asking and say sorry. If you are unsure about certain query, say sorry and advise the user to contact the instructor at instructor@ntu.edu.sg"
        ### any other things to add on

        ## Constructing initial system message
        sysmsg = f"{persona} {task} {context} {condition}"
        self.chat_history = [SystemMessage(sysmsg)]


        ## Built-in Vector Store / Database
        self.vector_store = retriever

    # A function taking user input to retrieve relavent context, consolidate the context, and pass to LLM. The function also store the user input and LLM response to chat_history. Finally, the function return the LLM response to the caller.
    def invoke(self, input):

        # Add user input to history
        self.chat_history.append(HumanMessage(content=input))

        if self.vector_store is not None:
            contexts = self.vector_store.get_context(input)
            input = f"Given the following chat history: {self.chat_history} along with new context: {contexts}, answer the user question: {input}"
        else:
            input = f"Given the following chat history: {self.chat_history}, answer the user question: {input}"

        # Call LLM
        response = self.llm.invoke(input)

        # Add LLM response to history
        self.chat_history.append(response)

        return response



### UNCOMMENT FOR LOCAL TESTING
# "C:/CDEFG/250117 - [Session 1, 2025] Fundamental of Context-Aware LLM-Based Chatbot Development (FYP Student, B2408)/Workshop 2 by Bernice Koh/.env"

# "C:/CDEFG/250122 - [Session 2, 2025] Getting started with Multi-Agents Chatbot Development and Deployment on Azure/Workshop 2 by Ong Chin Ann/vectorstores/sc1015"


# import os
# from dotenv import load_dotenv

# envpath = "<.env file path>"
# vector_store_path = "<local vector store path>"

# load_dotenv(dotenv_path=envpath, override="true")

# retriever = ContextRetriever(
#     endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
#     apikey=os.environ['AZURE_OPENAI_APIKEY'],
#     deployment_name=os.environ['AZURE_TEXT_EMBEDDING'],
#     model_name=os.environ['AZURE_TEXT_EMBEDDING_MODEL'],
#     vs_path=vector_store_path
# )

# llm = ChatEngine(
#     endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
#     apikey=os.environ['AZURE_OPENAI_APIKEY'],
#     deployment_name=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
#     model_name=os.environ['AZURE_OPENAI_MODEL_NAME'],
#     api_version=os.environ['AZURE_OPENAI_API_VERSION'],
#     retriever=retriever
# )

# rs = llm.invoke("Hello World, what is this course all about?")
# print(rs.content)