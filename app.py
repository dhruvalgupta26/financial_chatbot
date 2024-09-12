import streamlit as st
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.embeddings import OllamaEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

@st.cache_data
def process_csv(file_path):
    loader = CSVLoader(file_path=file_path)
    data = loader.load()
    return data

def split_data(data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(data)
    return splits

st.title("Finance Chatbot")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

# Main chat interface
chat_container = st.container()

# Initialize session state for messages and embeddings
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your Finance Chatbot. How can I assist you today?"}]
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

# Display chat messages
for message in st.session_state.messages:
    with chat_container:
        st.chat_message(message["role"]).write(message["content"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    
    with st.spinner("Processing CSV file..."):
        process_data = process_csv(temp_file_path)
        st.success("CSV file processed successfully!")

    with st.spinner("Splitting data into chunks..."):
        chunks = split_data(process_data)
        st.success("Data split into chunks successfully!")

    with st.spinner("initializing llm"):
        llm = ChatGroq(model="Llama-3.1-70b-Versatile")
    if st.session_state.embeddings is None:
        with st.spinner("Initializing embeddings..."):
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.success("Embeddings initialized successfully!")

    if st.session_state.vectordb is None:
        with st.spinner("Converting chunks into vector embeddings..."):
            st.session_state.vectordb = FAISS.from_documents(chunks, embedding=st.session_state.embeddings)
            st.success("Vector embeddings created successfully!")

    retriever = st.session_state.vectordb.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial analytics chatbot. Provide accurate and helpful answers based on your financial knowledge and the given context."),
        ("ai", "I understand. I'm here to assist with financial analysis and questions. How can I help you today?"),
        ("human", "{input}"),
        ("human", "Context: {context}"),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    def get_response(input_text):
        return retrieval_chain.invoke({"input": input_text, "chat_history": memory.chat_memory.messages})

    st.success("Chatbot is ready! You can now ask your questions.")

# User input
if user_input := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with chat_container:
        st.chat_message("user").write(user_input)

    if uploaded_file is not None:
        with st.spinner("Generating response..."):
            response = get_response(user_input)
            assistant_response = response['answer']
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            with chat_container:
                st.chat_message("assistant").write(assistant_response)

            # Update memory
            memory.chat_memory.add_user_message(user_input)
            memory.chat_memory.add_ai_message(assistant_response)
    else:
        st.warning("Please upload a CSV file first.")

# Display chat history
with st.expander("Chat History"):
    for message in st.session_state.messages:
        st.write(f"{message['role'].capitalize()}: {message['content']}")
