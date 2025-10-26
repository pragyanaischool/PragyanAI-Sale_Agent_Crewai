import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew, Process

# Import your custom RAG tool function
from tools.rag_manager import get_rag_tool

# --- Page Configuration ---
st.set_page_config(
    page_title="Sales PragyanAI Program",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("🤖 Agent Sales PragyanAI")
st.markdown("Upload a document, ask a question, and let the AI agents analyze it for you.")

# --- API Key Management (Sidebar) ---
with st.sidebar:
    st.header("🔑 API Configuration")
    st.markdown("""
    This app requires API keys for Groq (LLM) and MongoDB (Vector Store).
    
    1.  **Groq API Key**: Get from [GroqCloud](https://console.groq.com/keys)
    2.  **MongoDB URI**: Get from your [MongoDB Atlas](https://cloud.mongodb.com/) cluster.
    """)
    
    # Load .env file if it exists (for local development)
    if os.path.exists(".env"):
        load_dotenv()
        st.sidebar.success("Loaded API keys from .env file.")

    groq_api_key = os.environ.get("GROQ_API_KEY")
    mongo_uri = os.environ.get("MONGO_URI")

    if not groq_api_key:
        groq_api_key = st.text_input("Groq API Key", type="password", key="groq_key_input")
    else:
        st.sidebar.markdown(f"**Groq API Key**: `...{groq_api_key[-4:]}` (Loaded)")

    if not mongo_uri:
        mongo_uri = st.text_input("MongoDB Atlas URI", type="password", key="mongo_uri_input")
    else:
        st.sidebar.markdown(f"**MongoDB URI**: Loaded (hidden)")

    st.header("⚙️ RAG Configuration")
    DB_NAME = st.text_input("MongoDB DB Name", value="pragyan_ai_db")
    COLLECTION_NAME = st.text_input("MongoDB Collection Name", value="sales_docs")
    INDEX_NAME = st.text_input("MongoDB Vector Index", value="vector_index")
    st.caption(f"Remember to create a Vector Search Index in MongoDB named **`{INDEX_NAME}`** for the `embedding` field (384 dimensions).")

# --- Main Application UI ---
uploaded_file = st.file_uploader(
    "1. Upload your document (.pdf, .docx, .txt)",
    type=["pdf", "docx", "txt"]
)

user_question = st.text_input(
    "2. Ask a question about your document",
    placeholder="e.g., What are the key services mentioned?"
)

start_button = st.button("🚀 Analyze and Answer")

st.markdown("---")
st.subheader("🤖 Agent Answer:")

# Placeholder for the final answer
answer_placeholder = st.empty()
answer_placeholder.info("Your answer will appear here...")

# --- Main Execution Logic ---
if start_button:
    # 1. Validate all inputs
    if not uploaded_file:
        st.error("Please upload a document first.")
    elif not user_question:
        st.error("Please enter a question.")
    elif not groq_api_key:
        st.error("Please provide your Groq API Key in the sidebar.")
    elif not mongo_uri:
        st.error("Please provide your MongoDB URI in the sidebar.")
    else:
        try:
            # 2. Save the uploaded file to a temporary location
            temp_dir = "temp"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            temp_file_path = os.path.join(temp_dir, f"{int(time.time())}_{uploaded_file.name}")
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 3. Run the RAG pipeline and agent crew
            with st.status("🚀 **Agents are at work...**", expanded=True) as status:
                try:
                    status.write("Initializing Groq LLM...")
                    llm = ChatGroq(api_key=groq_api_key, model="mixtral-8x7b-32768")
                    
                    status.write(f"Reading & embedding document: `{uploaded_file.name}`...")
                    rag_tool = get_rag_tool(
                        file_path=temp_file_path,
                        mongo_uri=mongo_uri,
                        db_name=DB_NAME,
                        collection_name=COLLECTION_NAME,
                        index_name=INDEX_NAME
                    )

                    # Handle RAG tool creation error
                    if rag_tool.name == "Error Tool":
                        error_message = rag_tool._run(query="")
                        raise Exception(f"Failed to process document: {error_message}")
                    
                    status.write("Initializing agents and tasks...")
                    
                    # 4. Define Agents
                    document_analyst = Agent(
                        role="Document Analyst",
                        goal=f"Analyze the uploaded document to find the most relevant information to answer the user's question: '{user_question}'.",
                        backstory="You are an expert at parsing documents and extracting key information. You must use your 'document_query_tool' to find the answer.",
                        tools=[rag_tool],
                        llm=llm,
                        verbose=True,
                        allow_delegation=False
                    )
                    
                    sales_assistant = Agent(
                        role="Sales AI Assistant",
                        goal="Formulate a clear, concise, and helpful answer to the user's question based on the context provided by the Document Analyst.",
                        backstory="You are a friendly and professional AI assistant. You do not have access to any tools or documents yourself; you rely *only* on the context given to you to formulate your final answer.",
                        llm=llm,
                        verbose=True,
                        allow_delegation=False
                    )
                    
                    # 5. Define Tasks
                    analysis_task = Task(
                        description=f"Use your tool to query the document and find all relevant information that answers this question: **'{user_question}'**. Pass the full, un-summarized retrieved context to the Sales AI Assistant.",
                        expected_output="The raw, relevant text snippets retrieved from the document that directly answer the user's question.",
                        agent=document_analyst,
                    )
                    
                    response_task = Task(
                        description=f"Using the context provided by the Document Analyst, craft a final, human-readable answer to the user's question: **'{user_question}'**.",
                        expected_output="A clear and helpful answer to the user's question, based *only* on the provided context.",
                        agent=sales_assistant,
                        context=[analysis_task]  # This task depends on the output of the analysis_task
                    )
                    
                    # 6. Initialize and Run Crew
                    status.write("🚀 **Agents are running...**")
                    crew = Crew(
                        agents=[document_analyst, sales_assistant],
                        tasks=[analysis_task, response_task],
                        process=Process.sequential,
                        verbose=2
                    )
                    
                    final_result = crew.kickoff()
                    
                    status.update(label="✅ **Analysis Complete!**", state="complete", expanded=False)
                    
                    # 7. Display the result
                    answer_placeholder.empty() # Clear the "Your answer will appear here..."
                    st.markdown(final_result)
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    status.update(label="❌ **Error!**", state="error", expanded=True)
                
                finally:
                    # 8. Clean up the temporary file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
