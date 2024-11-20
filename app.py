import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
import os
import tempfile

def initialize_session_state():
    # Ensure session state contains required variables
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey!"]

def handle_conversation(query, chain, history):
    # Generate a response from the chain and update history
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_interface(chain):
    # Containers for chat history and user input
    reply_container = st.container()
    user_input_container = st.container()

    with user_input_container:
        with st.form(key='user_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your PDF", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                response = handle_conversation(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(response)

    if st.session_state['generated']:
        # Display the conversation history
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=f"{i}_user", avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

def setup_conversational_chain(vector_store):
    # Configure the language model
    llm = LlamaCpp(
        streaming=True,
        model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        temperature=0.75,
        top_p=1,
        verbose=True,
        n_ctx=4096
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create a conversational chain with the retriever and memory
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )

def main():
    # Set up session state variables
    initialize_session_state()

    st.title("Content Engine - Rahul Kumar")
    st.sidebar.title("Document Processing")

    # Upload and process documents
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

    if uploaded_files:
        all_texts = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]

            # Temporarily save uploaded file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_path)

            if loader:
                all_texts.extend(loader.load())
                os.remove(temp_path)

        # Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(all_texts)

        # Generate embeddings using a pre-trained model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2", 
            model_kwargs={'device': 'cpu'}
        )

        # Build a vector store for text retrieval
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        # Set up the conversational chain
        chain = setup_conversational_chain(vector_store)

        # Display the chat interface
        display_chat_interface(chain)

if __name__ == "__main__":
    main()
