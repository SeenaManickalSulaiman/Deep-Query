---
title: Deep Query
emoji: 🚀
colorFrom: green
colorTo: purple
sdk: streamlit
sdk_version: 1.41.1
app_file: app.py
pinned: false
---

# Deep Query
### Overview
This project is a demonstration of how to build an AI-powered Document Q&A System using Groq’s large language models (LLMs) integrated with LangChain’s RAG (Retrieval-Augmented Generation) framework. The app processes uploaded documents (PDF, DOCX, TXT, WhatsApp chats, and social media files—both structured and unstructured), enabling users to ask questions and receive accurate, context-aware answers directly from the content they’ve provided. The app ensures that answers are strictly grounded in the uploaded documents, and it transparently indicates when answers are not available—preventing hallucinations and maintaining reliability.

### Problem Statement

Traditional document Q&A systems face several challenges:
1. Poor Retrieval Quality: Incomplete or irrelevant passages can lead to incorrect or hallucinated answers.
2. Contextual Gaps: Answer generation often pulls unrelated information from outside the document, leading to misleading responses.
3. Lack of Trust: Users need confidence that answers are grounded in their own documents, especially for sensitive data like WhatsApp chats or social media content.

### Proposed Solution
This project introduces a robust, document-grounded Q&A system that:
* Parses a wide range of documents—including WhatsApp chats, social media exports, PDFs, DOCX, and TXT files—using reliable parsing libraries (PyMuPDF, python-docx, and text parsing).
* Utilizes LangChain’s RAG architecture with FAISS and Hugging Face embeddings to retrieve the most relevant content chunks.
* Fine-tunes Groq’s large language models on domain-specific tasks to enhance accuracy and relevance in answering questions from documents.
* Ensures that answers come strictly from uploaded documents, and if no relevant content is found, the system explicitly informs the user.
* Provides interactive analytics to monitor usage trends and query categories for continuous improvement.

### Features
** Document Versatility:**  Supports PDFs, DOCX, TXT, WhatsApp chats, and social media data—both structured and unstructured.
** Document-Grounded Answers: ** Answers are generated solely from uploaded documents, preventing hallucinations.
** Fine-Tuning Ready:**  Utilizes Groq’s fine-tuned LLMs for domain-specific Q&A.
** Fast Vector Search:**  Uses Hugging Face embeddings and FAISS for efficient context retrieval.
** Interactive Analytics:**  Provides dashboards showing query categories and usage trends.
** User-Friendly Interface:**  Built with Streamlit for easy uploads, configuration, and interactive querying.

## Componets of Deep Query
![image](https://github.com/user-attachments/assets/714a746e-b229-41ea-b412-33f3010054f8)




