import streamlit as st
import os
import openai
from pypdf import PdfReader
from llama_index.llms.openai import OpenAI
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import StorageContext

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from chromadb import PersistentClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration class
class Configs:
    db_dir = "./chroma_db"  # Directory for the persistent database
    db_name = "my_collection"  # Name of the ChromaDB collection
    open_API_Key = os.getenv("OPENAI_API_KEY")  # OpenAI API Key
    embedding_model = "text-embedding-ada-002"  # OpenAI embedding model
    batch_size = 32  # Embedding batch size


class VectorDBManager:
    def __init__(self):
        self.vector_store = None
        self.docstore = None
        self.embed_model = None

    def create_or_get_vector_db(self):
        '''Creates or gets the Chroma DB and instantiates Vector Store, Docstore, and Open AI embedding model'''
        # Creating a Chroma PersistentClient instance
        db = PersistentClient(path=Configs.db_dir)

        # Configure OpenAI embedding function
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=Configs.open_API_Key,
            model_name=Configs.embedding_model
        )

        # Get or create a collection
        chroma_collection = db.get_or_create_collection(Configs.db_name, embedding_function=openai_ef)

        # Initialize vector store, docstore, and embedding model
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self.docstore = SimpleDocumentStore()
        self.embed_model = OpenAIEmbedding(model=Configs.embedding_model, embed_batch_size=Configs.batch_size)

        return self.vector_store, self.docstore, self.embed_model


# Initialize VectorDBManager
vector_db_manager = VectorDBManager()

# Ensure the data directory exists
if not os.path.exists("data"):
    os.makedirs("data")

# Function to extract text from a PDF using PyPDF
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to split text into nodes
def split_text_into_nodes(text):
    splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=20,
    )
    nodes = splitter.get_nodes_from_documents([Document(text=text)])
    return nodes

# Function to embed and store data in vector DB index
def embedding_and_storing(nodes, vector_store, embed_model):
    # Convert nodes to Documents
    documents = [Document(text=node.text) for node in nodes]

    # Create index from documents (nodes)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)

    return index

# Function to retrieve results
def retrieve_results(query, index):
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return response.response

# Streamlit UI
def main():
    st.title("Chatbot using RAG - POC")

    # API key input
    openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")

    if openai_api_key:
        Configs.open_API_Key = openai_api_key
        st.session_state["openai_api_key"] = openai_api_key
        if 'hide_api_success' not in st.session_state:
            st.session_state['hide_api_success'] = False

        if not st.session_state['hide_api_success']:
            st.success("API key saved successfully ✅")
            st.session_state['hide_api_success'] = True  # Hide the success message when the upload starts

        # File uploader
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Extract text from PDF
            with st.spinner("Extracting text from PDF..."):
                text = extract_text_from_pdf(file_path)

            # Split text into nodes
            with st.spinner("Splitting text into nodes..."):
                nodes = split_text_into_nodes(text)

            # Initialize VectorDB
            with st.spinner("Initializing VectorDB..."):
                vector_store, docstore, embed_model = vector_db_manager.create_or_get_vector_db()

            # Embed and store in vector DB index
            with st.spinner("Embedding text and storing in vector index..."):
                index = embedding_and_storing(nodes, vector_store, embed_model)

            # Chat with the indexed documents
            query = st.text_input("Ask a question about the document:")
            if query:
                with st.spinner("Retrieving results..."):
                    results = retrieve_results(query, index)

                st.write(results)

if __name__ == "__main__":
    main()
