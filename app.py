# Deep Eye with Groq: File-based Question Answering System
import streamlit as st
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
import docx  # python-docx

# Updated LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Supported models (as per Groq docs)
SUPPORTED_MODELS = [
    "llama3-8b-8192",    # Current recommended default
    "llama3-70b-8192",   # Larger model for complex tasks
    "mixtral-8x7b-32768" # Still available as of latest docs
]

# File processing functions
def load_txt(file):
    return file.read().decode("utf-8")

def load_pdf(file):
    pdf_data = ""
    with fitz.open(stream=file.read(), filetype="pdf") as pdf:
        for page in pdf:
            pdf_data += page.get_text()
    return pdf_data

def load_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def process_uploaded_file(file):
    file_extension = file.name.split('.')[-1].lower()
    if file_extension == "txt":
        return load_txt(file)
    elif file_extension == "pdf":
        return load_pdf(file)
    elif file_extension == "docx":
        return load_docx(file)
    return None

def setup_vector_store(file_content):
    # Split text with improved text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_text(file_content)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create and store vector store
    st.session_state.vector_store = FAISS.from_texts(texts, embeddings)
    return st.session_state.vector_store.as_retriever()

def main():
    st.set_page_config(
        page_title="Deep Eye with Groq",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Deep Eye with Groq")
    
    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Configuration")
        model = st.selectbox(
            "Choose a model",
            SUPPORTED_MODELS,
            index=0,
            help="Select from Groq's currently supported models"
        )
        
        st.markdown("---")
        st.markdown("### How to use:")
        st.markdown("1. Upload a document (TXT/PDF/DOCX)")
        st.markdown("2. Ask questions about the content")
        st.markdown("3. Get instant AI-powered answers")
        
        st.markdown("---")
        st.markdown("Made with ❤️ using [Groq](https://groq.com/)")
    
    # File upload
    uploaded_file = st.file_uploader(
        "📄 Upload a document (TXT, PDF, or DOCX)", 
        type=["txt", "pdf", "docx"]
    )
    
    # Process file if uploaded
    if uploaded_file and st.session_state.vector_store is None:
        with st.spinner("Processing document..."):
            file_content = process_uploaded_file(uploaded_file)
            if file_content:
                setup_vector_store(file_content)
                st.success("Document processed and ready for questions!")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Initialize Groq chat (done here to respect model selection)
    try:
        groq_chat = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=model,
            temperature=0.3  # More deterministic answers for Q&A
        )
    except Exception as e:
        st.error("Failed to initialize Groq client. Please check your API key.")
        st.stop()
    
    # Chat input
    if prompt := st.chat_input("Ask about your document..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process question if we have a document
        if st.session_state.vector_store:
            try:
                # Create prompt template
                prompt_template = ChatPromptTemplate.from_template("""
                Answer the question based only on the provided context.
                Be concise and accurate. If unsure, say you don't know.
                
                Context: {context}
                
                Question: {input}
                """)
                
                # Create document chain
                document_chain = create_stuff_documents_chain(
                    groq_chat, 
                    prompt_template
                )
                
                # Create and execute retrieval chain
                retriever = st.session_state.vector_store.as_retriever()
                qa_chain = create_retrieval_chain(retriever, document_chain)
                
                with st.spinner("Thinking..."):
                    response = qa_chain.invoke({"input": prompt})
                    answer = response["answer"]
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(answer)
                
                # Add to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as e:
                st.error(f"Error processing your question: {str(e)}")
        else:
            st.warning("Please upload a document first.")

if __name__ == "__main__":
    main()