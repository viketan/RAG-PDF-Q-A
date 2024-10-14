import streamlit as st
import os
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq.chat_models import ChatGroq
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Langsmith Tracking (set up environment variables for tracking)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG app(PDF)"

def get_vectorstore(uploaded_file):
    # Save the uploaded file to a temporary directory
    if uploaded_file is not None:
        temp_file_path = os.path.join("temp", uploaded_file.name)
        
        # Create temp directory if not exists
        os.makedirs("temp", exist_ok=True)

        # Save the uploaded file
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Now pass the saved file path to PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        doc = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap= 1000)
        chunk_doc = splitter.split_documents(doc)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(documents=chunk_doc,embedding=embeddings)
        return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatGroq(model="gemma-7b-It", temperature=0.3)
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatGroq(model="gemma-7b-It", temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    try:
        response = conversation_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input,
        })
        return response['answer']
    except Exception as e:
        st.error("The response was blocked due to safety concerns.")
        return "The response was blocked due to safety concerns."


def main():
    st.title("Q&A with PDF")

    with st.sidebar:
        st.header("Settings")
        pdf = st.file_uploader("Please upload the PDF",type="pdf")

    if not pdf:
            st.info("Please upload the PDF")
            return
        
    if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello I'm personal assistant. What queries do you regarding PDF that you have uploaded")
            ]

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore(pdf)

    user_query = st.chat_input("Type your message here...")
    if user_query:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

if __name__ == "__main__":
    main()