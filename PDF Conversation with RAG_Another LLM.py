## RAG with conversation with PDF including chat history

import streamlit as st
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableWithMessageHistory
import os


#Langsmith tracking

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

#Load the environment variables from .env file like Keys to access Langchain, Huggingface and Groq.

from dotenv import load_dotenv
load_dotenv()

# Selection of Word embedding model - converting the chunks into vecor embeddings using the Hugging face embeddings.

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

st.title("Chatbot with PDF files using conversational RAG")
st.write("Upload the PDF and chat with the content")

# This key is required to access the open source LLM models hosted on Groq platform - https://console.groq.com/docs/models
# Groq provides LPU ( Language Processing Unit) as compared to CPU or GPU to provder fast AI inference.
api_key = st.text_input("Enter your Groq API",type="password")

## check if groq API key is provided

if api_key:
    llm = ChatGroq(groq_api_key=api_key,model_name="llama3-groq-70b-8192-tool-use-preview")

    #Chat interface
    session_id = st.text_input("Session ID",value="default")
    ## Statefully manage the chat history
    if 'store' not in st.session_state:
        st.session_state.store={}

    #Upload files
    upload_files = st.file_uploader("Choose A PDF file to upload",type="pdf",accept_multiple_files=True)
    ## Process uploaded files
    if upload_files:
        documents=[]
        for uploaded_file in upload_files:
            temppdf = f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        #Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        # Store the embeddings in vector store db - Chroma
        vectorstore = Chroma.from_documents(documents=splits,embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Prompt
        contextulize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat hisotry"
            "formulate a standalone question which can be understood"
            "without the chat history, DO NOT answet the question"
            "just reformulate it if needed and otherwise return as is."
        )
        contextulize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",contextulize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(llm,retriever,contextulize_q_prompt)

        ## Answer question prompt
        system_prompt = (
            "You are an assistant for question answering tasks"
            "Use the following piece of retrieved context to answer"
            "the question. If you don't know the answer say that you"
            "dont know. Use three sentences maximum and keep the "
            "answer concise"
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system",system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human","{input}"),

                    ]
            )
        question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)
        
        def get_session_history(session:str)-> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]    

        conversationl_rag_chain = RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversationl_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                },
            ) 
            st.write(st.session_state.store)
            st.write("Assistant:",response['answer'])
            st.write("Chat history:",session_history.messages)

else:
    st.warning("Please enter the GROQ API key")