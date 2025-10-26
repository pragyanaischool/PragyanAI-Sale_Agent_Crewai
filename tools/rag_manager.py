import os
from pymongo import MongoClient
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
# Updated import: Removed BaseTool, added create_retriever_tool
from langchain_core.tools import Tool, create_retriever_tool
from typing import Any

# Import your local tool
from .document_reader_tool import DocumentReaderTool

# --- 1. Define Input Schema ---
# No longer needed, as create_retriever_tool handles this automatically.


# --- 2. The RAG Tool Logic ---

def get_rag_tool(
    file_path: str,
    mongo_uri: str,
    db_name: str,
    collection_name: str,
    index_name: str
) -> Tool:
    """
    This function:
    1. Reads a document from a file path.
    2. Chunks the document text.
    3. Embeds the chunks using Hugging Face.
    4. Stores them in a MongoDB Atlas Vector Search collection.
    5. Creates and returns a LangChain Tool (retriever) for querying that collection.
    """
    
    # --- Step 1: Read the Document ---
    try:
        reader = DocumentReaderTool()
        doc_text = reader.run(file_path=file_path)
        
        # We need to wrap the raw text in a LangChain 'Document' object
        # for the text splitter to work.
        documents = [Document(page_content=doc_text)]
        
    except Exception as e:
        error_message = f"Error reading document: {e}"
        print(f"DEBUG: {error_message}")
        return Tool(
            name="Error Tool",
            # Use *args, **kwargs to accept any input and still return the error
            func=lambda *args, **kwargs: error_message,
            description="Returns an error message.",
        )

    # --- Step 2: Chunk the Document ---
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150
        )
        doc_chunks = text_splitter.split_documents(documents)
    except Exception as e:
        error_message = f"Error splitting document: {e}"
        print(f"DEBUG: {error_message}")
        return Tool(
            name="Error Tool",
            func=lambda *args, **kwargs: error_message,
            description="Returns an error message.",
        )

    # --- Step 3: Embed and Store in MongoDB ---
    try:
        # Initialize Hugging Face embeddings
        # This uses the 'all-MiniLM-L6-v2' model, which is fast and local.
        # It creates 384-dimensional vectors.
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Connect to MongoDB
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        
        # Clear existing documents for this specific file/session if needed
        # For simplicity, we'll just add new ones.
        # In a production app, you might add metadata to track the source file.
        
        # Create the MongoDBAtlasVectorSearch instance and add documents
        # This will embed and upload the chunks in one go.
        vector_store = MongoDBAtlasVectorSearch.from_documents(
            documents=doc_chunks,
            embedding=embeddings,
            collection=collection,
            index_name=index_name,
        )
        
    except Exception as e:
        error_message = f"Error connecting to/embedding in MongoDB: {e}. Check MONGO_URI, and ensure your vector index '{index_name}' is set for 384 dimensions."
        print(f"DEBUG: {error_message}")
        return Tool(
            name="Error Tool",
            func=lambda *args, **kwargs: error_message,
            description="Returns an error message.",
        )

    # --- Step 4: Create Retriever Tool ---
    try:
        # Create a retriever from the vector store
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5} # Retrieve top 5 relevant chunks
        )
        
        # Use the new, simpler create_retriever_tool function
        # This replaces the manual 'retrieve_and_format' function
        document_query_tool = create_retriever_tool(
            retriever,
            "document_query_tool",
            "Use this tool to query the uploaded document. Pass the user's question as the 'query' argument."
        )
        
        return document_query_tool
        
    except Exception as e:
        error_message = f"Error creating retriever tool: {e}"
        print(f"DEBUG: {error_message}")
        return Tool(
            name="Error Tool",
            func=lambda *args, **kwargs: error_message,
            description="Returns an error message.",
        )

