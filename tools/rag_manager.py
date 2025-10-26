import os
from pymongo import MongoClient
from langchain_community.document_loaders.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.tools.retriever import create_retriever_tool
from crewai_tools import BaseTool

# Import your custom document reader
from tools.document_reader_tool import DocumentReaderTool

def get_rag_tool(file_path: str, mongo_uri: str, db_name: str, collection_name: str, index_name: str) -> BaseTool:
    """
    Manages the entire RAG pipeline and returns a CrewAI tool for querying.
    
    1. Reads the document using DocumentReaderTool.
    2. Splits the text into manageable chunks.
    3. Initializes Hugging Face embeddings.
    4. Connects to MongoDB Atlas Vector Search.
    5. Clears any old data for this document (optional, but good for demos).
    6. Embeds and stores the new document chunks.
    7. Creates and returns a retriever tool for the CrewAI agent.
    
    Args:
        file_path: The local path to the user-uploaded document.
        mongo_uri: Connection string for MongoDB Atlas.
        db_name: Name of the MongoDB database.
        collection_name: Name of the MongoDB collection.
        index_name: Name of the Vector Search index in MongoDB Atlas.

    Returns:
        A CrewAI-compatible BaseTool for semantic searching.
    """
    
    # --- 1. Read Document ---
    print(f"[RAG Manager] Reading document from: {file_path}")
    reader_tool = DocumentReaderTool()
    document_text = reader_tool._run(file_path=file_path)
    
    if document_text.startswith("Error:"):
        print(f"[RAG Manager] Error reading document: {document_text}")
        # Return a "dummy" tool that just reports the error
        class ErrorTool(BaseTool):
            name: str = "Error Tool"
            description: str = "A dummy tool that reports a document reading error."
            def _run(self, query: str) -> str:
                return document_text
        return ErrorTool()

    # Wrap the text in a LangChain Document object
    documents = [Document(page_content=document_text, metadata={"source": file_path})]

    # --- 2. Split Text ---
    print("[RAG Manager] Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    
    if not docs:
        print("[RAG Manager] Error: No text could be split from the document.")
        # Return a dummy error tool
        class ErrorTool(BaseTool):
            name: str = "Error Tool"
            description: str = "A dummy tool that reports a document splitting error."
            def _run(self, query: str) -> str:
                return "Error: Could not split document into text chunks."
        return ErrorTool()

    # --- 3. Initialize Embeddings ---
    print("[RAG Manager] Initializing Hugging Face embeddings (all-MiniLM-L6-v2)...")
    # Using a popular, fast, and lightweight model (384 dimensions)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'} # Use CPU
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    # --- 4. Connect to MongoDB ---
    print(f"[RAG Manager] Connecting to MongoDB Atlas: {db_name}.{collection_name}")
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    
    # --- 5. Clear Old Data (Optional) ---
    # For this demo, we'll clear the collection each time a new file is uploaded
    # In a real app, you might manage this differently (e.g., based on user ID or file ID)
    print(f"[RAG Manager] Clearing old data from collection...")
    collection.delete_many({})

    # --- 6. Embed and Store ---
    print(f"[RAG Manager] Embedding and storing {len(docs)} chunks in MongoDB...")
    vector_search = MongoDBAtlasVectorSearch.from_documents(
        documents=docs,
        embedding=embeddings,
        collection=collection,
        index_name=index_name
    )
    
    # --- 7. Create Retriever Tool ---
    print("[RAG Manager] Creating retriever tool...")
    retriever = vector_search.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5} # Return top 5 most relevant chunks
    )
    
    # Use LangChain's helper to create a tool
    document_query_tool = create_retriever_tool(
        retriever,
        "document_query_tool",
        "A tool to query the uploaded document. Use this to find specific information, facts, or answers from the document's content."
    )
    
    print("[RAG Manager] RAG tool successfully created.")
    return document_query_tool

# Example of how this module would be used (for testing)
# if __name__ == "__main__":
#     # This requires a .env file with your keys
#     from dotenv import load_dotenv
#     load_dotenv()
    
#     MONGO_URI = os.getenv("MONGO_URI")
#     DB_NAME = "pragyan_ai_db"
#     COLLECTION_NAME = "sales_docs"
#     INDEX_NAME = "vector_index" # Your Atlas Vector Search index name
    
#     # Create a dummy test.txt file
#     with open("test_rag.txt", "w") as f:
#         f.write("The sky is blue. The grass is green. CrewAI is a framework for AI agents.")
        
#     print("--- Testing RAG Manager ---")
#     rag_tool = get_rag_tool(
#         file_path="test_rag.txt",
#         mongo_uri=MONGO_URI,
#         db_name=DB_NAME,
#         collection_name=COLLECTION_NAME,
#         index_name=INDEX_NAME
#     )
    
#     print(f"\nTool Created: {rag_tool.name}")
#     print(f"Tool Description: {rag_tool.description}")
    
#     # Test the tool
#     query = "What color is the sky?"
#     result = rag_tool._run(query)
#     print(f"\n--- Testing Tool with Query: '{query}' ---")
#     print(result)
    
#     # Clean up
#     os.remove("test_rag.txt")
