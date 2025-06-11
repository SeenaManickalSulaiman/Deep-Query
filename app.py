# app.py

import streamlit as st
import os
from PIL import Image
import fitz  # PyMuPDF
import docx  # python-docx
import pandas as pd
import plotly.express as px

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

# Supported models
SUPPORTED_MODELS = [
    "llama3-8b-8192",
    "llama3-70b-8192"
]

# Load images
try:
    header_image = Image.open("header_image.jpg")
    sidebar_image = Image.open("sidebar_image.png")
except Exception as e:
    header_image = None
    sidebar_image = None
    st.warning("Header/sidebar images not found. Please upload header_image.jpg and sidebar_image.png.")

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
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.split_text(file_content)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        st.session_state.vector_store = FAISS.from_texts(texts, embeddings)
        st.session_state.document_processed = True
        return True
    except Exception as e:
        st.error(f"Error setting up vector store: {str(e)}")
        return False

def main():
    st.set_page_config(page_title="Deep Query with Groq", page_icon="🧠", layout="wide")
    
    # Tabs
    tab1, tab2 = st.tabs(["📝 Ask Deep Query", "📊 Analytics"])

    # Custom CSS (optional)
    st.markdown("""
        <style>
        .css-18e3th9 {padding-top: 0px;}
        .css-1d391kg {padding-top: 0px;}
        .css-1u6zpdt {padding-left: 0px;}
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        if sidebar_image:
            st.image(sidebar_image, caption="Deep Query: Your AI Powered Document Assistant", use_container_width=True)
        st.title("⚙️ Configuration")
        model = st.selectbox(
            "Choose a model",
            SUPPORTED_MODELS,
            index=0,
            help="Select from Groq's supported models"
        )
        st.markdown("---")
        st.markdown("### How to use:")
        st.markdown("1. Upload a document (TXT/PDF/DOCX).")
        st.markdown("2. Ask summary or questions about the document.")
        st.markdown("3. Get AI-powered answers.")
        st.markdown("---")
        st.markdown("Made with ❤️ using [Groq](https://groq.com/)")

    # Tab 1: Ask Deep Query
    with tab1:
        if header_image:
            st.image(header_image, use_container_width=True)
        st.header("Deep Query")
        uploaded_file = st.file_uploader("Upload a document:", type=["txt", "pdf", "docx"])

        if uploaded_file and not st.session_state.document_processed:
            with st.spinner("Processing document..."):
                file_content = process_uploaded_file(uploaded_file)
                if file_content:
                    if setup_vector_store(file_content):
                        st.success("Document processed and ready for questions!")
                    else:
                        st.error("Failed to process document.")

        prompt = st.text_input("Ask a question about your document:")
        if st.button("Submit"):
            if not st.session_state.vector_store:
                st.warning("Please upload and process a document first.")
            elif prompt.strip():
                groq_api_key = os.environ.get("GROQ_API_KEY")
                if not groq_api_key:
                    st.error("GROQ_API_KEY not set. Please set it in your Space's environment variables.")
                    st.stop()
                try:
                    groq_chat = ChatGroq(
                        groq_api_key=groq_api_key,
                        model_name=model,
                        temperature=0.3
                    )
                    prompt_template = ChatPromptTemplate.from_template("""
                    Answer the question based only on the provided context.
                    Be concise and accurate. If unsure, say you don't know.

                    Context: {context}

                    Question: {input}
                    """)
                    document_chain = create_stuff_documents_chain(groq_chat, prompt_template)
                    retriever = st.session_state.vector_store.as_retriever()
                    qa_chain = create_retrieval_chain(retriever, document_chain)

                    with st.spinner("Thinking..."):
                        response = qa_chain.invoke({"input": prompt})
                        answer = response["answer"]
                    
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.success("Query processed successfully!")
                    st.markdown("### 💡 Deep Eye Says")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error during Q&A: {str(e)}")
            else:
                st.warning("Please enter a question.")

    # Tab 2: Analytics
    with tab2:
        st.header("📊 Analytics Dashboard")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px;">
                <h4 style="color: #4caf50;">Document Categories</h4>
                """,
                unsafe_allow_html=True,
            )
            # Placeholder data
            df_queries = pd.DataFrame({
                "category": ["General", "Technical", "Legal"],
                "Count": [5, 3, 2]
            })
            fig_query = px.bar(
                df_queries,
                x="category",
                y="Count",
                color="category",
                text="Count"
            )
            st.plotly_chart(fig_query, use_container_width=True)
            st.markdown(
                """
                **Insight:** This chart shows how many queries fall into each document category. 
                It helps identify which document types or topics are most frequently explored.
                """
            )

        with col2:
            st.markdown(
                """
                <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px;">
                <h4 style="color: #ff5722;">Engagement Over Time</h4>
                """,
                unsafe_allow_html=True,
            )
            df_engagement = pd.DataFrame({
                "year_month": ["2025-01", "2025-02", "2025-03"],
                "Count": [10, 15, 8]
            })
            df_engagement["year_month"] = pd.Categorical(df_engagement["year_month"], ordered=True)
            fig_engagement = px.line(
                df_engagement,
                x="year_month",
                y="Count",
                markers=True,
                labels={"year_month": "Month-Year", "Count": "Number of Questions"}
            )
            st.plotly_chart(fig_engagement, use_container_width=True)
            st.markdown(
                """
                **Insight:** This chart displays the number of questions asked per month 
                . It helps understand usage trends and see 
                when users are most active on the platform.
                """
            )

    # Footer
    st.markdown("""
        ---
        Developed by [Seena MS](https://www.linkedin.com/in/seenams/)
    """)

if __name__ == "__main__":
    main()
