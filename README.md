# Content-Engine-for-Document-Comparison-and-Insights
This repository contains the implementation of a Content Engine that utilizes Retrieval Augmented Generation (RAG) techniques for analyzing and comparing multiple PDF documents, specifically Form 10-K filings from multinational companies.
Content Engine for Document Comparison and Insights
This repository contains the implementation of a Content Engine that utilizes Retrieval Augmented Generation (RAG) techniques for analyzing and comparing multiple PDF documents, specifically Form 10-K filings from multinational companies.

Features
PDF Parsing: Extracts and processes text from PDF documents.
Vectorization: Uses local embedding models to convert text content into vectors.
Vector Store Integration: Stores embeddings in a vector store (Chroma or FAISS) for fast retrieval and comparison.
Local LLM Integration: Runs a local Large Language Model (LLM) for generating insights and answering queries.
Interactive Chatbot Interface: Built using Streamlit, allowing users to query the system and obtain insights from the documents.
Workflow
Parse Documents: Extract text from provided PDF documents (Alphabet, Tesla, and Uber Form 10-K filings).
Generate Embeddings: Use a local embedding model to create vectors for document content.
Store in Vector Store: Store the document embeddings in a vector store like Chroma or FAISS.
Query Engine: Retrieve relevant documents and generate insights based on user queries.
Chatbot Interface: Users can interact with the system through a Streamlit-based chatbot UI.
Sample Queries
"What are the risk factors associated with Google and Tesla?"
"What is the total revenue for Google Search?"
"What are the differences in the business of Tesla and Uber?"
Setup Instructions
To run the project locally:

Clone the repository.
Install required dependencies with pip install -r requirements.txt.
Run the Streamlit app: streamlit run streamlit_app.py.
Project Structure
streamlit_app.py: Streamlit app for the interactive UI.
pdf_parsing.py: Functions for extracting text from PDFs.
vector_store.py: Code for storing and querying document embeddings.
llm_integration.py: Integrates a local LLM for generating insights.
requirements.txt: List of dependencies.
Technologies Used
LlamaIndex or LangChain (for retrieval-augmented generation)
Streamlit (for the interactive UI)
ChromaDB or FAISS (for storing document embeddings)
Sentence-Transformers (for text embeddings)
PyPDF2 (for PDF parsing)
