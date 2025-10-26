Agent Sales PragyanAI Program

This is a Streamlit application that uses CrewAI to answer questions about user-uploaded documents.

It implements a full Retrieval-Augmented Generation (RAG) pipeline:

File Upload: Accepts PDF, DOCX, and TXT files via the Streamlit UI.

Document Processing: Uses crewai-tools and pypdf to read and extract text.

Vectorization: Splits the text and uses a Hugging Face model (sentence-transformers/all-MiniLM-L6-v2) to create vector embeddings locally and for free.

Storage: Stores the vectors in MongoDB Atlas Vector Search.

Agentic Crew: Uses a CrewAI team (powered by Groq's Llama 3) to:

Analyze the user's question.

Query the vector database to find relevant document chunks.

Synthesize a final, helpful answer.

Setup & Running

Clone the repository:

git clone [your-repo-url]
cd Sales-PragyanAI-App


Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


(Note: The first time you run the app, sentence-transformers will download the embedding model, which may take a minute.)

Set up API Keys:

Create a .env file for local development.

Create a .streamlit/secrets.toml file for deployment.

Add your keys for:

GROQ_API_KEY (from GroqCloud)

MONGO_ATLAS_URI (from MongoDB Atlas)

MongoDB Atlas Setup:

Create a Database named pragyan_ai and a Collection (e.g., sales_docs).

In Atlas, go to the Search tab and create a Vector Search Index on your collection named vector_index.

Important: Your index definition must match the dimensions of the embedding model. For all-MiniLM-L6-v2, the dimension is 384.

Your JSON index definition should look something like this:

{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    }
  ]
}


Run the app:

streamlit run app.py
