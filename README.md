# Content Engine for Document Comparison and Insights

This repository implements a **Content Engine** utilizing **Retrieval Augmented Generation (RAG)** techniques to analyze and compare multiple PDF documents, specifically Form 10-K filings from multinational companies.

## Features

- **PDF Parsing**: Extracts and processes text from PDF documents.
- **Vectorization**: Converts text content into vectors using local embedding models.
- **Vector Store Integration**: Embeddings are stored in a vector store (Chroma or FAISS) for efficient retrieval and comparison.
- **Local LLM Integration**: A local Large Language Model (LLM), **Mistral-7B-Instruct**, is used for generating insights and answering queries.
- **Interactive Chatbot Interface**: Built with **Streamlit**, providing users a platform to query the system and obtain document insights.

---

## Workflow

1. **Parse Documents**: Extract text from PDF files (e.g., Alphabet, Tesla, and Uber Form 10-K filings).
2. **Generate Embeddings**: Use **Sentence-Transformers** to generate dense vector embeddings for document text.
3. **Store in Vector Store**: Save embeddings in a vector store like **Chroma** or **FAISS** for fast querying.
4. **Query Engine**: Retrieve relevant information and generate insights using a **local LLM**.
5. **Chatbot Interface**: Users interact via a Streamlit-based chatbot UI to query and explore insights from documents.

---

## Sample Queries
- "How does Tesla's automotive segment differ from its energy generation and storage segment?"
- "What are the differences in the business of Tesla and Uber?"
- "What is the total revenue for Google Search?"

---

## Setup Instructions

To run the project locally:

1. Clone the repository.
2. Install required dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## Project Structure

- **`streamlit_app.py`**: Streamlit app for the interactive user interface.
- **`pdf_parsing.py`**: Functions to extract and preprocess text from PDFs.
- **`vector_store.py`**: Code for storing and querying document embeddings.
- **`llm_integration.py`**: Integrates the **Mistral-7B-Instruct** LLM for generating responses.
- **`requirements.txt`**: List of dependencies for the project.

---

## Technologies Used

- **LlamaIndex** or **LangChain**: For retrieval-augmented generation (RAG) tasks.
- **Streamlit**: Interactive UI for user interaction.
- **ChromaDB** or **FAISS**: Vector store for document embeddings.
- **Sentence-Transformers**: Text embeddings generation.
- **PyPDF2**: For PDF parsing.
- **Mistral-7B-Instruct**: Local LLM used for query responses and insights generation.
