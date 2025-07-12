import os
import streamlit as st
from qdrant_client import QdrantClient
from urllib.parse import urlparse
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer

from src.load_file import load_file, read_file
from src.load_doc_vector import upload_faq_vector
from src.logo_title import logo_title
from src.process_faq import process_faqs
from src.load_doc_vector import upload_document

# Main execution block
if __name__ == "__main__":
    # Initialize session state variables if they don't exist
    if 'vectordb_collection_name' not in st.session_state:
        st.session_state.vectordb_collection_name = "default_collection"  # Default collection name
    if 'qdrant_client' not in st.session_state:
        # Default Qdrant client configuration with specified host and port
        st.session_state.qdrant_client = None
    if 'embed_model' not in st.session_state:
        st.session_state.embed_model = "all-MiniLM-L6-v2"  # Default embedding model
    if 'chunk_size' not in st.session_state:
        st.session_state.chunk_size = 1000  # Default chunk size
    if 'chunk_overlap' not in st.session_state:
        st.session_state.chunk_overlap = 200  # Default chunk overlap
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.1  # Default temperature for language model

    # Sidebar for temperature configuration
    st.sidebar.header("Model Configuration")
    st.session_state.temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Adjust the temperature for the language model (lower values make responses more deterministic, higher values increase creativity)."
    )

    logo_title("🦜🔗 Golden FAQ Generation With Variability, Similarity Score and Answer Generation.")
    st.write("""
In this application, FAQ generation is automated by processing uploaded documents to extract key questions and answers, using language models for validation and paraphrasing to ensure variability in the responses. These FAQs are then converted into embeddings and stored in a vector database (e.g., Qdrant, Pinecone) for efficient semantic search. 
When a user queries the system, it performs a nearest-neighbor search using similarity metrics like cosine similarity, retrieving the most relevant FAQs and generating accurate, context-aware answers. 
This enables fast, scalable, and contextually relevant responses even for complex or rephrased queries.
""")

    # Qdrant connection configuration
    st.subheader("Qdrant Server Configuration")
    qdrant_host = st.text_input("Qdrant Host", value="4.240.112.22", help="Enter the Qdrant server host (e.g., 4.240.112.22)")
    qdrant_port = st.number_input("Qdrant Port", value=6333, help="Enter the Qdrant server port (default: 6333)")
    
    # Initialize Qdrant client with user-provided settings
    try:
        st.session_state.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        # Test connection by listing collections
        st.session_state.qdrant_client.get_collections()
        st.success("Successfully connected to Qdrant server!")
    except Exception as e:
        st.error(f"Failed to connect to Qdrant server at {qdrant_host}:{qdrant_port}. Error: {str(e)}")
        st.error("Please ensure the Qdrant server is running and the host/port are correct.")
        st.stop()  # Halt execution until connection is fixed

    # Qdrant Vector database client 
    collection_name_faq = 'faq_' + st.session_state.vectordb_collection_name
    st.session_state.collection_name_faq = collection_name_faq
    collection_name_doc = st.session_state.vectordb_collection_name

    qdrant_client = st.session_state.qdrant_client
    # Create or check the FAQ collection
    try:
        qdrant_client.get_collection(collection_name_faq)
    except Exception:
        embedding_dimensions = SentenceTransformer(st.session_state.embed_model).get_sentence_embedding_dimension()
        qdrant_client.create_collection(
            collection_name=collection_name_faq, 
            vectors_config=VectorParams(
                size=embedding_dimensions,  # Vector size is defined by used model
                distance=Distance.COSINE
            )
        )
    # Create or check the document collection
    try:
        st.session_state.qdrant_client.get_collection(collection_name_doc)
    except Exception:
        qdrant_client.create_collection(
            collection_name=collection_name_doc,
            vectors_config=VectorParams(
                size=SentenceTransformer(st.session_state.embed_model).get_sentence_embedding_dimension(),  # Vector size is defined by used model
                distance=Distance.COSINE
            )
        )

    with st.form("my_form"):
        prompt_faq = st.text_area("Enter the FAQ prompt", value=
        f"""You are an AI assistant tasked with generating a list of potential questions that a citizen might ask about the services in the context. 
These questions should be varied, relevant, and comprehensive, covering all aspects of the services as provided in the context.
Create questions of varying complexity and phrasing to address different levels of understanding among citizens.
Include the service name in question to enhance semantic search accuracy.
Provide simple non-nested questions.
Provide only the questions and avoid any additional text. 
Present the questions directly, without any additional text.
        """, height=300)

        prompt_faq_variation = st.text_area("Enter the prompt for FAQ variation", value=
        f"""System: You are an AI assistant assigned to rephrase a citizen's question about the UP government services.
Your task is to generate diverse rephrased versions of the original question while maintaining its original intent and ensuring each rephrased question aligns with the provided context.
        
User: Generate multiple variations of the original question that maintain the same meaning and relevance. 
Provide only the rephrased questions, ensuring each one includes the service name for improved semantic search accuracy. 
Ensure that all rephrased questions maintain the intent of the original query and are directly relevant to the context provided.
Provide only the rephrased questions and avoid any additional text.
           """, height=300)
        
        # Upload multiple files
        uploaded_files = st.file_uploader("Choose multiple files", type=["txt", "pdf", "csv", "json", "html", "xls", "docx"], accept_multiple_files=True)
 
        # Option to enter a URL
        url_input = st.text_input("Or Enter a website URL to load")

        # Bulk File Upload
        bulk_file_name = st.file_uploader("Or Choose bulk file name (.txt)", type=["txt"], accept_multiple_files=False)

        submitted = st.form_submit_button("Submit")
            
        if submitted:
            if uploaded_files:
                # Process each file one by one
                for idx, file in enumerate(uploaded_files):
                    st.subheader(f"File {idx + 1}: {file.name}")
                    
                    # Define a temporary file path (use platform-agnostic temp directory)
                    file_path = os.path.join(os.path.expanduser("~"), "temp", file.name)
                    st.write(file_path)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create directory if it doesn't exist

                    # Write the uploaded file to the temporary path
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())

                    if file_path:
                        upload_document(file_path, st.session_state.vectordb_collection_name, st.session_state.qdrant_client, st.session_state.embed_model, st.session_state.chunk_size, st.session_state.chunk_overlap, file.name)
                    else:
                        st.write("File name is empty.")
                    
                    # Create document vector database
                    try:
                        st.session_state.qdrant_client.get_collection(collection_name_doc)
                    except Exception:
                        qdrant_client.create_collection(
                            collection_name=collection_name_doc,
                            vectors_config=VectorParams(
                                size=SentenceTransformer(st.session_state.embed_model).get_sentence_embedding_dimension(),  # Vector size is defined by used model
                                distance=Distance.COSINE
                            )
                        )

                    results_df = process_faqs(file_path, prompt_faq, prompt_faq_variation)  # Creating Golden FAQ & its variability
                    upload_faq_vector(results_df, file_path, st.session_state.qdrant_client, st.session_state.embed_model, collection_name_faq)  # Uploading the FAQ to vector db
                    st.session_state.qdrant_client.delete_collection(collection_name=collection_name_doc)  # Delete the document collection
            elif url_input:
                file_content = load_file(url=url_input)
                parsed_url = urlparse(url_input)
                path = parsed_url.path
                filename = os.path.basename(path)
                file_path = os.path.join(os.path.expanduser("~"), "temp", filename)
                st.write(f"File loaded from URL: {url_input}")
                st.text_area("Content", file_content, height=300)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create directory if it doesn't exist
                with open(file_path, "wb") as f:
                    f.write(file_content.getbuffer())
                if file_path:
                    upload_document(file_path, st.session_state.vectordb_collection_name, st.session_state.qdrant_client, st.session_state.embed_model, st.session_state.chunk_size, st.session_state.chunk_overlap, filename)
                else:
                    st.write("File name is empty.")
                
                # Create document vector database
                try:
                    st.session_state.qdrant_client.get_collection(collection_name_doc)
                except Exception:
                    qdrant_client.create_collection(
                        collection_name=collection_name_doc,
                        vectors_config=VectorParams(
                            size=SentenceTransformer(st.session_state.embed_model).get_sentence_embedding_dimension(),  # Vector size is defined by used model
                            distance=Distance.COSINE
                        )
                    )

                results_df = process_faqs(file_path, prompt_faq, prompt_faq_variation)  # Creating Golden FAQ & its variability
                upload_faq_vector(results_df, file_path, st.session_state.qdrant_client, st.session_state.embed_model, collection_name_faq)  # Uploading the FAQ to vector db
                st.session_state.qdrant_client.delete_collection(collection_name=collection_name_doc)  # Delete the document collection
            elif bulk_file_name:
                try: 
                    # Define a temporary file path
                    file_path = os.path.join(os.path.expanduser("~"), "temp", bulk_file_name.name)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create directory if it doesn't exist

                    # Write the uploaded file to the temporary path
                    with open(file_path, "wb") as f:
                        f.write(bulk_file_name.getbuffer())

                    file_path = read_file(file_path)
                    file_paths = file_path.split('\n')

                    for file_path in file_paths:
                        if file_path != "":
                            # Create document vector database
                            try:
                                st.session_state.qdrant_client.get_collection(collection_name_doc)
                            except Exception:
                                qdrant_client.create_collection(
                                    collection_name=collection_name_doc,
                                    vectors_config=VectorParams(
                                        size=SentenceTransformer(st.session_state.embed_model).get_sentence_embedding_dimension(),  # Vector size is defined by used model
                                        distance=Distance.COSINE
                                    )
                                )

                            upload_document(file_path, st.session_state.vectordb_collection_name, st.session_state.qdrant_client, st.session_state.embed_model, st.session_state.chunk_size, st.session_state.chunk_overlap, file_path)
                            results_df = process_faqs(file_path, prompt_faq, prompt_faq_variation)  # Creating Golden FAQ & its variability
                            upload_faq_vector(results_df, file_path, st.session_state.qdrant_client, st.session_state.embed_model, collection_name_faq)  # Uploading the FAQ to vector db
                            st.session_state.qdrant_client.delete_collection(collection_name=collection_name_doc)  # Delete the document collection
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.write("File name is empty.")