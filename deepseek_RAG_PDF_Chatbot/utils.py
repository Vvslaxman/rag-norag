# Import required libraries
import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# Function to load, split, and embed data from PDF documents into Chroma vector store
def process_documents(pdfs):
    """
    Process PDF documents through loading, splitting, and embedding.
    Returns vector store instance.
    """
    # Create temporary directory for PDF storage
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded PDFs to temp directory
        pdf_paths = []
        for pdf in pdfs:
            path = os.path.join(temp_dir, pdf.name)
            with open(path, "wb") as f:
                f.write(pdf.getbuffer())
            pdf_paths.append(path)
        
        # Load the documents
        documents = []
        for path in pdf_paths:
            loader = PDFPlumberLoader(path)
            documents.extend(loader.load())
        
        # Split documents into chunks using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,  
            chunk_overlap=150  
        )
        splits = text_splitter.split_documents(documents)
        
        # Instantiate the embeddings model
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Create embeddings and vector store
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        return vector_store

# Initialize and returns a retriever for the vector store, which will be used to fetch relevant chunks from the stored embeddings based on user queries. 
def get_retriever():
    """Initialize and return the vector store retriever"""
    # Initialize the embedding model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    try:
        # Initialize the vector store
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )

        # Return the retriever with MMR (Maximum Marginal Relevance) search and k=3
        return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})

    except Exception as e:
        print(f"Error initializing vector store: {e}")
        return None


