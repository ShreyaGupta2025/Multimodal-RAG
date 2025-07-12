import streamlit as st
from src.logo_title import logo_title

st.set_page_config(page_title="Multimodal RAG Demo", layout="wide")
logo_title("Multimodal RAG Demo")
# st.title("Welcome to the Multimodal RAG Demo")
st.markdown(
    """
    ## About This Application

    This application demonstrates a **Multimodal Retrieval-Augmented Generation (RAG) System** that processes PDF files to extract and summarize text, tables, and images. It leverages advanced language models and embedding techniques to create a searchable document index stored in Qdrant, a vector database.

    ### Key Features:
    - **PDF Processing:** Upload one or more PDF files to extract text, tables, and images.
    - **Summarization:** Generate summaries for text chunks and tables optimized for semantic retrieval.
    - **Image Analysis:** Extract and summarize images, ensuring that even visual content is captured.
    - **Qdrant Storage:** Processed content is stored in Qdrant for fast and efficient similarity-based retrieval.
    - **User-Friendly Interface:** A wide layout and organized sections make it easy to interact with the system.

    ### How It Works:
    1. **Upload PDFs:** On the upload page, you can select one or more PDFs. The system will extract the content from these files, create summaries, and display a clubbed version of the original text.
    2. **Process and Store:** The extracted information is processed and stored in a Qdrant vector database, enabling efficient retrieval based on user queries.
    3. **Retrieve Information:** On the retrieval page, you can ask a question, and the system will use its internal retriever to find the most relevant pieces of information and generate an answer.

    This demo is designed to showcase the integration of various modern NLP and document processing tools into a single, unified application. Whether you're looking to explore how AI can enhance document search or simply want to see advanced PDF processing in action, this application has something for you.

    ---
    Enjoy exploring the Multimodal RAG Demo!
    """
)
