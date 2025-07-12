import os
import logging
import streamlit as st
import time
from urllib.parse import quote
import re
import json

import torch
import pickle
import uuid

import fitz  # pymupdf
import io
import base64
import openai
import streamlit.components.v1 as components

from dotenv import load_dotenv
from functools import lru_cache
from rank_bm25 import BM25Okapi

from langchain_openai import AzureChatOpenAI
from langchain.embeddings.base import Embeddings

from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever

from FlagEmbedding import BGEM3FlagModel

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


from src.logo_title import logo_title
from sentence_transformers import SentenceTransformer

load_dotenv()  

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Streamlit Page Setup
st.set_page_config(page_icon=":shark:", layout="wide", initial_sidebar_state="expanded")
logo_title("Chatbot")

# ─── CONFIG ─────────────────────────────────────────────────────────────
CANDIDATE_POOL = 20
FINAL_K = 10
BM25_PICKLE = "bm25_index.pkl"
BM25_METADATA = "bm25_metadata.pkl"
COLLECTION_NAME = "AIS_DOPT"
FAQ_COLLECTION_NAME = "user_queries_faq"

PDF_BASE_PATH = r"/Users/rajendragupta/Documents/Python/RAG/documents/AIS Rules"

AVAILABLE_MODELS = [
    "gpt-4o-mini",
    "gpt-4.1",
    "DeepSeek-R1",
    "Llama-4-Maverick-17B-128E-Instruct",
    "Meta-Llama-3.1-405B-Instruct",
]

MODEL_DISPLAY_MAPPING = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4.1": "gpt-4.1",
    "DeepSeek-R1": "DeepSeek-R1",
    "Llama-4-Maverick-17B-128E-Instruct": "Llama-4",
    "Meta-Llama-3.1-405B-Instruct": "Llama-3-405B",
}

MODEL_DISPLAY_NAMES = [MODEL_DISPLAY_MAPPING[model] for model in AVAILABLE_MODELS]
MODEL_REVERSE_MAPPING = {display_name: backend_name for backend_name, display_name in MODEL_DISPLAY_MAPPING.items()}

# Initialize FAQ embedding model at startup
faq_embedder = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("Initialized FAQ embedder: all-MiniLM-L6-v2")

def initialize_chat_model(selected_model):
    """
    Initialize the chat model based on the selected model name.
    Returns AzureChatOpenAI instance for OpenAI models or openai.OpenAI client for SambaNova models.
    """
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    sambanova_api_key = os.getenv("SAMBANOVA_API_KEY")

    if selected_model in ["gpt-4o-mini", "gpt-4.1"]:
        if not azure_api_key or not azure_endpoint:
            logger.error("Azure credentials missing: AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not found in .env")
            st.error("Azure API key or endpoint is missing. Please add AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT to your .env file.")
            return None
        try:
            logger.info(f"Initializing AzureChatOpenAI model: {selected_model}")
            return AzureChatOpenAI(
                model=selected_model,
                temperature=0.1,
                streaming=True,
                openai_api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                api_version="2025-01-01-preview"
            )
        except Exception as e:
            logger.error(f"Failed to initialize AzureChatOpenAI model {selected_model}: {str(e)}")
            st.error(f"Failed to initialize Azure model: {str(e)}")
            return None
    elif selected_model in [
        "DeepSeek-R1",
        "Llama-4-Maverick-17B-128E-Instruct",
        "Meta-Llama-3.1-405B-Instruct",
    ]:
        if not sambanova_api_key:
            logger.error("SAMBANOVA_API_KEY not found in .env")
            st.error("SambaNova API key is missing. Please add SAMBANOVA_API_KEY to your .env file.")
            return None
        try:
            logger.info(f"Initializing SambaNova client for model: {selected_model}")
            return openai.OpenAI(
                api_key=sambanova_api_key,
                base_url="https://api.sambanova.ai/v1"
            )
        except Exception as e:
            logger.error(f"Failed to initialize SambaNova client for {selected_model}: {str(e)}")
            st.error(f"Failed to initialize SambaNova model: {str(e)}")
            return None
    else:
        logger.error(f"Unsupported model: {selected_model}")
        st.error(f"Unsupported model: {selected_model}")
        return None

def create_pdf_link(pdf_name: str, page: int, link_text: str = None) -> str:
    """
    Build an HTML <a> tag that opens the given PDF at the desired page within Streamlit.
    """
    if link_text is None:
        link_text = f"{pdf_name} (Page {page})"
    # Ensure .pdf extension is not duplicated
    pdf_name_clean = pdf_name.rsplit(".pdf", 1)[0] if pdf_name.endswith(".pdf") else pdf_name
    return f'<a href="#" onclick="openPDF(\'{pdf_name_clean}\', {page});">{link_text}</a>'

class BGEM3Embeddings(Embeddings):
    def __init__(self, model_name="BAAI/bge-m3", device='cpu', use_fp16=False, batch_size=32, max_length=8192):
        self.model = None
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self.max_length = max_length

    def _load_model(self):
        if self.model is None:
            self.model = BGEM3FlagModel(self.model_name, device=self.device, use_fp16=self.use_fp16)
            logger.info(f"Loaded BGEM3FlagModel: {self.model_name}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        outputs = self.model.encode(texts, batch_size=self.batch_size, max_length=self.max_length)
        return outputs["dense_vecs"].tolist()

    def embed_query(self, text: str) -> list[float]:
        self._load_model()
        outputs = self.model.encode([text], batch_size=1, max_length=self.max_length)
        return outputs["dense_vecs"][0].tolist()

@st.cache_resource
def init_embeddings():
    return BGEM3Embeddings(
        model_name="BAAI/bge-m3",
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_fp16=True,
        batch_size=32,
        max_length=8192
    )

@st.cache_resource
def init_reranker():
    return HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_resource
def init_qdrant_client():
    return QdrantClient(
        url="http://localhost:6333",
        timeout=30,  # Increased timeout to handle network delays
        prefer_grpc=False
    )

@st.cache_resource
def init_vectorstore(_client, _embeddings):
    return QdrantVectorStore(
        client=_client,
        collection_name=COLLECTION_NAME,
        embeddings=_embeddings
    )

# Initialize FAQ collection
@st.cache_resource
def initialize_faq_collection(_client):
    try:
        _client.get_collection(FAQ_COLLECTION_NAME)
        logger.info(f"FAQ collection {FAQ_COLLECTION_NAME} already exists")
    except Exception as e:
        logger.warning(f"Failed to get FAQ collection: {str(e)}. Creating new collection...")
        _client.create_collection(
            collection_name=FAQ_COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        logger.info(f"Created FAQ collection {FAQ_COLLECTION_NAME}")

# Store query in FAQ collection
def store_faq_query(query, response, user_id="anonymous"):
    query_vector = faq_embedder.encode(query).tolist()
    point_id = str(uuid.uuid4())
    payload = {
        "query": query,
        "response": response,
        "timestamp": time.time(),
        "user_id": user_id
    }
    client.upsert(
        collection_name=FAQ_COLLECTION_NAME,
        points=[PointStruct(id=point_id, vector=query_vector, payload=payload)]
    )

# Search similar queries in FAQ collection
def search_faq_queries(query, top_k=1, threshold=0.9):
    query_vector = faq_embedder.encode(query).tolist()
    search_result = client.search(
        collection_name=FAQ_COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        score_threshold=threshold
    )
    return search_result

embeddings = init_embeddings()
reranker_model = init_reranker()
client = init_qdrant_client()
vectorstore = init_vectorstore(client, embeddings)
initialize_faq_collection(client)

@lru_cache(maxsize=1000)
def cached_similarity_search(query_str: str, k: int = FINAL_K):
    query_str = query_str.lower().strip()
    return vectorstore.similarity_search(
        query_str,
        k=k,
        search_params={"exact": False, "hnsw_ef": 32}
    )

def load_all_documents(collection_name=COLLECTION_NAME, batch_size=100):
    all_docs = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        if not records:
            break
        for rec in records:
            payload = rec.payload if hasattr(rec, "payload") else rec.get("payload", {})
            content = payload.get("page_content", "")
            if content.strip():
                all_docs.append(
                    Document(
                        page_content=content,
                        metadata=payload.get("metadata", {}),
                        id=rec.id
                    )
                )
        if not offset:
            break
    return all_docs

@st.cache_data
def build_and_cache_bm25():
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        current_doc_count = collection_info.points_count
        current_timestamp = collection_info.config.params.vectors.get('size', 0)
    except Exception as e:
        logger.warning(f"Failed to get collection info: {e}. Rebuilding BM25 index...")
        current_doc_count, current_timestamp = 0, 0

    cached_metadata = {'doc_count': 0, 'timestamp': 0}
    if os.path.exists(BM25_METADATA):
        try:
            with open(BM25_METADATA, "rb") as f:
                cached_metadata = pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load BM25 metadata: {e}")

    if (os.path.exists(BM25_PICKLE) and
        cached_metadata['doc_count'] == current_doc_count and
        cached_metadata['timestamp'] == current_timestamp):
        try:
            with open(BM25_PICKLE, "rb") as f:
                bm25_index, docs = pickle.load(f)
            logger.info("Loaded BM25 index from cache")
            return bm25_index, docs
        except Exception as e:
            logger.warning(f"Failed to load BM25 index: {e}. Rebuilding...")

    logger.info("Building BM25 index...")
    docs = load_all_documents()
    tokenized = [doc.page_content.lower().split() for doc in docs]
    bm25_index = BM25Okapi(tokenized, k1=1.2, b=0.75)
    with open(BM25_PICKLE, "wb") as f:
        pickle.dump((bm25_index, docs), f)
    with open(BM25_METADATA, "wb") as f:
        pickle.dump({'doc_count': current_doc_count, 'timestamp': current_timestamp}, f)
    logger.info("BM25 index built and cached")
    return bm25_index, docs

bm25_index, bm25_docs = build_and_cache_bm25()

@st.cache_resource
def init_bm25_retriever():
    return BM25Retriever.from_documents(bm25_docs, k=CANDIDATE_POOL)

bm25_retriever = init_bm25_retriever()

def build_basic_retriever(top_k=FINAL_K):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": top_k,
            "search_params": {"exact": False, "hnsw_ef": 32}
        }
    )

def build_hybrid_retriever(top_k=FINAL_K):
    qdrant_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": CANDIDATE_POOL,
            "search_params": {"exact": False, "hnsw_ef": 32}
        }
    )
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, qdrant_retriever],
        weights=[0.5, 0.5]
    )
    reranker_compressor = CrossEncoderReranker(model=reranker_model, top_n=top_k)
    return ContextualCompressionRetriever(
        base_retriever=ensemble_retriever,
        base_compressor=reranker_compressor
    )

def build_rag_prompt(question, retrieved_docs):
    context_text = ""
    citations = []
    for idx, doc in enumerate(retrieved_docs, 1):
        file_name = doc.metadata.get("file_name", "Unknown")
        page_number = doc.metadata.get("page_number", "Unknown")
        if file_name == "Unknown" or page_number == "Unknown":
            logger.warning(f"Invalid metadata for doc {idx}: file_name={file_name}, page_number={page_number}")
            continue
        try:
            page_number = int(page_number)
        except (ValueError, TypeError):
            logger.warning(f"Invalid page_number for doc {idx}: {page_number}")
            continue
        citation = {"source": file_name, "page_number": page_number}
        citations.append(citation)
        if doc.metadata.get("is_image"):
            snippet = f"\n[Image summary {idx}]\nSummary: {doc.page_content}\nCitations: {json.dumps(citation)}\n---\n"
        else:
            summary_content = doc.page_content
            table_markdown = doc.metadata.get("table_markdown", "")
            combined = f"Summary: {summary_content}\nTable Markdown: {table_markdown}\n" if table_markdown else summary_content + "\n"
            snippet = f"\n[Context {idx}]\n{combined}Citations: {json.dumps(citation)}\n---\n"
        context_text += snippet
    if not context_text:
        context_text = "No relevant information found in the provided documents."
    return (
        f"You are JankariSetu, a friendly chatbot for a government in India. Answer the user's question based on the provided context. "
        f"Include citations for EVERY piece of information used, in the format [Source: {{file_name}}, Page: {{page_number}}]. Each citation should"
        f"be unique with page number all page number should not be in the single list. "
        f"If no relevant information is found, say 'I am not sure, please check if the document is inserted or provide more context.' "
        f"Use simple Indian-accented English and avoid apostrophes (e.g., 'is not' instead of 'isnt'). "
        f"Ensure all citations are included, even if the information is from multiple documents.\n\n"
        f"Context:\n{context_text}\n"
        f"Question: {question}\n\n"
        f"Provide a concise answer with appropriate citations. "
    )

def build_conversational_prompt(user_input):
    return (
        f"You are Jankarisetu, a friendly chatbot for a government in India. Respond to the users input: {user_input}\n\n"
        f"Guidelines:\n"
        f"- Use simple Indian-accented English.\n"
        f"- Do not use apostrophes (e.g., 'does not' instead of 'doesnt').\n"
        f"- Be polite and professional, suitable for government stakeholders.\n"
        f"- If the input is unclear, say: I am not sure what do you mean. Can you please provide more context?"
    )

def display_pdf(pdf_name: str, page_number: int, chunk_text: str = None):
    """
    Display the PDF within the Streamlit app using pdf.js.
    """
    try:
        # Ensure .pdf extension is not duplicated
        pdf_name_clean = pdf_name.rsplit(".pdf", 1)[0] if pdf_name.endswith(".pdf") else pdf_name
        pdf_path = os.path.join(PDF_BASE_PATH, f"{pdf_name_clean}.pdf")
        if not os.path.exists(pdf_path):
            logger.warning(f"PDF not found: {pdf_path}")
            st.error(f"PDF not found: {pdf_path}")
            return

        # Open PDF and read content
        pdf_document = fitz.open(pdf_path)
        page_number = max(0, min(page_number - 1, pdf_document.page_count - 1))

        # Save PDF to a temporary buffer
        output_buffer = io.BytesIO()
        pdf_document.save(output_buffer)
        pdf_document.close()
        pdf_data = output_buffer.getvalue()
        pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
        output_buffer.close()

        # HTML and JavaScript for pdf.js viewer
        pdf_viewer_html = f"""
        <div style="width: 100%; height: 800px;">
            <div id="pdf-viewer"></div>
        </div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.9.359/pdf.min.js"></script>
        <script>
            pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.9.359/pdf.worker.min.js';
            const base64PDF = "{pdf_base64}";
            const binaryString = atob(base64PDF);
            const len = binaryString.length;
            const bytes = new Uint8Array(len);
            for (let i = 0; i < len; i++) {{
                bytes[i] = binaryString.charCodeAt(i);
            }}
            const loadingTask = pdfjsLib.getDocument(bytes);
            loadingTask.promise.then(pdf => {{
                pdf.getPage({page_number + 1}).then(page => {{
                    const viewport = page.getViewport({{ scale: 1.0 }});
                    const canvas = document.createElement('canvas');
                    canvas.style.display = "block";
                    canvas.style.margin = "auto";
                    const context = canvas.getContext('2d');
                    canvas.height = viewport.height;
                    canvas.width = viewport.width;
                    document.getElementById('pdf-viewer').appendChild(canvas);
                    page.render({{ canvasContext: context, viewport: viewport }});
                }});
            }});
        </script>
        """
        components.html(pdf_viewer_html, height=850)
        logger.info(f"Displayed PDF: {pdf_name_clean}, Page: {page_number + 1}")
    except Exception as e:
        logger.error(f"Error displaying PDF {pdf_name}, Page {page_number}: {str(e)}")
        st.error(f"Failed to display PDF: {str(e)}")

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        {"role": "assistant", "content": "Hello! I am JankariSetu, your assistant for government data. "
                                         "You can chat with me or use '@query ' to ask about government data (e.g., '@query What is the budget?'). "
                                         "How can I help you today?"}
    ]
if "retriever_type" not in st.session_state:
    st.session_state["retriever_type"] = "Hybrid"
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = "gpt-4.1"
if "retrieved_docs" not in st.session_state:
    st.session_state["retrieved_docs"] = []
if "citation_tuples" not in st.session_state:
    st.session_state["citation_tuples"] = []
if "citation_pdfs" not in st.session_state:
    st.session_state["citation_pdfs"] = []
if "message_citations" not in st.session_state:
    st.session_state["message_citations"] = {}
if "selected_source" not in st.session_state:
    st.session_state["selected_source"] = {}
if "faq_mode" not in st.session_state:
    st.session_state["faq_mode"] = False

# Sidebar for Retriever, Model Selection, and FAQ Mode
with st.sidebar:
    st.header("Settings")
    retriever_choice = st.selectbox(
        "Select Retriever Type for @query",
        ["Hybrid", "Basic"],
        index=0 if st.session_state["retriever_type"] == "Hybrid" else 1
    )
    if retriever_choice != st.session_state["retriever_type"]:
        st.session_state["retriever_type"] = retriever_choice

    current_display_name = MODEL_DISPLAY_MAPPING[st.session_state["selected_model"]]
    
    model_display_choice = st.selectbox(
        "Select Chat Model",
        MODEL_DISPLAY_NAMES,
        index=MODEL_DISPLAY_NAMES.index(current_display_name)
    )
    model_choice = MODEL_REVERSE_MAPPING[model_display_choice]
    if model_choice != st.session_state["selected_model"]:
        st.session_state["selected_model"] = model_choice
        logger.info(f"Selected model changed to: {model_choice}")

    faq_mode = st.checkbox("Enable FAQ Mode", value=st.session_state["faq_mode"])
    if faq_mode != st.session_state["faq_mode"]:
        st.session_state["faq_mode"] = faq_mode
        logger.info(f"FAQ mode {'enabled' if faq_mode else 'disabled'}")

# JavaScript to handle PDF link clicks
st.markdown("""
<script>
function openPDF(pdfName, pageNumber) {
    const event = new CustomEvent('openPDF', { detail: { pdfName: pdfName, pageNumber: pageNumber } });
    window.dispatchEvent(event);
}
</script>
""", unsafe_allow_html=True)

# Chat Interface
chat_container = st.container()
input_container = st.container()

# Display Chat History with Citations
with chat_container:
    for idx, message in enumerate(st.session_state["chat_history"]):
        if message["role"] == "user" or (message["role"] == "assistant" and message["content"]):
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    if idx not in st.session_state["message_citations"]:
                        st.session_state["message_citations"][idx] = []

                    def replace_citation(match):
                        file_name = match.group(1).strip()
                        page_number = match.group(2).strip()
                        try:
                            page_number = int(page_number)
                            if page_number < 1:
                                page_number = 1
                                logger.warning(f"Adjusted invalid page number to 1 in citation: {match.group(0)}")
                        except (ValueError, TypeError):
                            page_number = 1
                            logger.warning(f"Invalid page number in citation: {match.group(0)}")
                        citation_tuple = (file_name, page_number)
                        if citation_tuple not in st.session_state["citation_tuples"]:
                            st.session_state["citation_tuples"].append(citation_tuple)
                        if citation_tuple not in st.session_state["message_citations"][idx]:
                            st.session_state["message_citations"][idx].append(citation_tuple)
                        pdf_key = f"{file_name}_{page_number}_{idx}"
                        if not any(ci.get("pdf_key") == pdf_key for ci in st.session_state["citation_pdfs"]):
                            chunk_text = None
                            for doc in st.session_state.get("retrieved_docs", []):
                                if (doc.metadata.get("file_name") == file_name and
                                    str(doc.metadata.get("page_number")) == str(page_number)):
                                    chunk_text = doc.metadata.get("raw_text")
                                    break
                            st.session_state["citation_pdfs"].append({
                                "message_idx": idx,
                                "file_name": file_name,
                                "page_number": page_number,
                                "pdf_key": pdf_key,
                                "chunk_text": chunk_text
                            })
                            logger.info(f"Stored PDF info for {file_name}, page {page_number}, message_idx {idx}, key {pdf_key}")
                        link_text = f"[Source: {file_name}, Page: {page_number}]"
                        # Ensure .pdf extension is not duplicated
                        pdf_name = file_name.rsplit(".pdf", 1)[0] if file_name.endswith(".pdf") else file_name
                        pdf_link = create_pdf_link(pdf_name, page_number, link_text)
                        logger.info(f"Generated PDF link: {pdf_link}")
                        return pdf_link

                    logger.debug(f"Processing response: {message['content']}")
                    formatted_content = re.sub(
                        r'\[Source:\s*([^,]+),\s*Page:\s*(\d+)\]',
                        replace_citation,
                        message["content"]
                    )
                    st.markdown(formatted_content, unsafe_allow_html=True)

                    if idx in st.session_state["message_citations"] and st.session_state["message_citations"][idx]:
                        st.markdown("**Citations:**")
                        for file_name, page_number in sorted(st.session_state["message_citations"][idx], key=lambda x: (x[0], x[1])):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                button_key = f"citation_button_{idx}_{file_name}_{page_number}"
                                if st.button(f"View {file_name}, Page {page_number}", key=button_key):
                                    st.session_state["selected_source"][idx] = (file_name, page_number)
                                    logger.info(f"Selected source for message_idx {idx}: {file_name}, Page {page_number} via citation button")
                            with col2:
                                chunk_key = f"chunk_button_{idx}_{file_name}_{page_number}"
                                if st.button("Show Chunk", key=chunk_key):
                                    for ci in st.session_state["citation_pdfs"]:
                                        if (ci["message_idx"] == idx and
                                            ci["file_name"] == file_name and
                                            ci["page_number"] == page_number):
                                            st.session_state["selected_source"][idx] = (file_name, page_number)
                                            st.session_state["show_chunk"] = ci.get("chunk_text", "No chunk text available")
                                            logger.info(f"Showing chunk for {file_name}, Page {page_number}")
                                            break

                    # Display chunk text if selected
                    if "show_chunk" in st.session_state and idx in st.session_state["selected_source"]:
                        st.markdown("**Retrieved Chunk:**")
                        st.text_area("Chunk Text", st.session_state["show_chunk"], height=100, key=f"chunk_text_{idx}")

                    # Display PDF viewer if a source is selected
                    if idx in st.session_state["selected_source"]:
                        file_name, page_number = st.session_state["selected_source"][idx]
                        chunk_text = None
                        for ci in st.session_state["citation_pdfs"]:
                            if (ci["message_idx"] == idx and
                                ci["file_name"] == file_name and
                                ci["page_number"] == page_number):
                                chunk_text = ci.get("chunk_text")
                                break
                        st.markdown(f"**Viewing: {file_name}, Page {page_number}**")
                        display_pdf(file_name, page_number, chunk_text)
                else:
                    st.markdown(message["content"], unsafe_allow_html=True)

# Chat Input
with input_container:
    user_input = st.chat_input("Chat or use '@query ' to ask about government data:")

# Process User Input
if user_input:
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    current_message_idx = len(st.session_state["chat_history"])
    st.session_state["citation_pdfs"] = [
        ci for ci in st.session_state["citation_pdfs"]
        if ci["message_idx"] != current_message_idx
    ]
    if current_message_idx in st.session_state["message_citations"]:
        del st.session_state["message_citations"][current_message_idx]
    if current_message_idx in st.session_state["selected_source"]:
        del st.session_state["selected_source"][current_message_idx]
    if "show_chunk" in st.session_state:
        del st.session_state["show_chunk"]
    logger.info(f"Cleared PDFs, citations, selected source, and chunk for message_idx {current_message_idx}")
    st.rerun()

# Process Assistant Response
if st.session_state["chat_history"] and st.session_state["chat_history"][-1]["role"] == "user":
    user_input = st.session_state["chat_history"][-1]["content"]
    selected_model = st.session_state["selected_model"]
    chat_model = initialize_chat_model(selected_model)

    if not chat_model:
        st.session_state["chat_history"].append({"role": "assistant", "content": "Error: Unable to initialize the chat model. Please check your API keys and model selection."})
        st.rerun()

    if user_input.lower().strip() in ["hello", "hi", "hey"] or len(user_input.strip()) < 2:
        answer = "Hello! Nice to meet you! How can I help you today?"
        st.session_state["chat_history"].append({"role": "assistant", "content": ""})
        with chat_container:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                streamed_text = ""
                for char in answer:
                    streamed_text += char
                    placeholder.markdown(streamed_text, unsafe_allow_html=True)
                    time.sleep(0.01)
                st.session_state["chat_history"][-1]["content"] = streamed_text
    else:
        if user_input.lower().startswith("@query ") and st.session_state["faq_mode"]:
            query_text = user_input[7:].strip()
            if not query_text:
                answer = "Please provide a question after @query. For example: @query What is the budget?"
                st.session_state["chat_history"].append({"role": "assistant", "content": ""})
                with chat_container:
                    with st.chat_message("assistant"):
                        placeholder = st.empty()
                        streamed_text = ""
                        for char in answer:
                            streamed_text += char
                            placeholder.markdown(streamed_text, unsafe_allow_html=True)
                            time.sleep(0.01)
                        st.session_state["chat_history"][-1]["content"] = streamed_text
            else:
                # Check FAQ collection first
                faq_results = search_faq_queries(query_text)
                if faq_results:
                    top_result = faq_results[0]
                    answer = f"FAQ Response: {top_result.payload['response']} (Matched Query: {top_result.payload['query']}, Similarity: {top_result.score:.2f})"
                    st.session_state["chat_history"].append({"role": "assistant", "content": ""})
                    with chat_container:
                        with st.chat_message("assistant"):
                            placeholder = st.empty()
                            streamed_text = ""
                            for char in answer:
                                streamed_text += char
                                placeholder.markdown(streamed_text, unsafe_allow_html=True)
                                time.sleep(0.01)
                            st.session_state["chat_history"][-1]["content"] = streamed_text
                else:
                    # Fallback to RAG pipeline
                    retriever = build_hybrid_retriever() if st.session_state["retriever_type"] == "Hybrid" else build_basic_retriever()
                    try:
                        retrieved_docs = retriever.get_relevant_documents(query_text)
                        st.session_state["retrieved_docs"] = retrieved_docs
                        logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query_text}")
                        prompt = build_rag_prompt(query_text, retrieved_docs)
                        st.session_state["chat_history"].append({"role": "assistant", "content": ""})
                        with chat_container:
                            with st.chat_message("assistant"):
                                placeholder = st.empty()
                                streamed_text = ""
                                if selected_model in ["gpt-4o-mini", "gpt-4.1"]:
                                    for chunk in chat_model.stream([HumanMessage(content=prompt)]):
                                        streamed_text += chunk.content
                                        placeholder.markdown(streamed_text, unsafe_allow_html=True)
                                        time.sleep(0.01)
                                else:
                                    messages = [
                                        {"role": "system", "content": "You are Jankarisetu, a friendly chatbot for a government in India. Use simple Indian-accented English and avoid apostrophes."},
                                        {"role": "user", "content": prompt}
                                    ]
                                    stream = chat_model.chat.completions.create(
                                        messages=messages,
                                        model=selected_model,
                                        temperature=0.1,
                                        top_p=0.1,
                                        stream=True
                                    )
                                    for chunk in stream:
                                        if chunk.choices[0].delta.content is not None:
                                            streamed_text += chunk.choices[0].delta.content
                                            placeholder.markdown(streamed_text, unsafe_allow_html=True)
                                            time.sleep(0.01)
                                st.session_state["chat_history"][-1]["content"] = streamed_text
                                logger.info(f"LLM response: {streamed_text}")
                                # Store in FAQ collection
                                store_faq_query(query_text, streamed_text)
                    except Exception as e:
                        error_msg = f"Error retrieving answer: {str(e)}"
                        logger.error(error_msg)
                        st.session_state["chat_history"].append({"role": "assistant", "content": error_msg})
        else:
            if user_input.lower().startswith("@query "):
                query_text = user_input[7:].strip()
                if not query_text:
                    answer = "Please provide a question after @query. For example: @query What is the budget?"
                    st.session_state["chat_history"].append({"role": "assistant", "content": ""})
                    with chat_container:
                        with st.chat_message("assistant"):
                            placeholder = st.empty()
                            streamed_text = ""
                            for char in answer:
                                streamed_text += char
                                placeholder.markdown(streamed_text, unsafe_allow_html=True)
                                time.sleep(0.01)
                            st.session_state["chat_history"][-1]["content"] = streamed_text
                else:
                    retriever = build_hybrid_retriever() if st.session_state["retriever_type"] == "Hybrid" else build_basic_retriever()
                    try:
                        retrieved_docs = retriever.get_relevant_documents(query_text)
                        st.session_state["retrieved_docs"] = retrieved_docs
                        logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query_text}")
                        prompt = build_rag_prompt(query_text, retrieved_docs)
                        st.session_state["chat_history"].append({"role": "assistant", "content": ""})
                        with chat_container:
                            with st.chat_message("assistant"):
                                placeholder = st.empty()
                                streamed_text = ""
                                if selected_model in ["gpt-4o-mini", "gpt-4.1"]:
                                    for chunk in chat_model.stream([HumanMessage(content=prompt)]):
                                        streamed_text += chunk.content
                                        placeholder.markdown(streamed_text, unsafe_allow_html=True)
                                        time.sleep(0.01)
                                else:
                                    messages = [
                                        {"role": "system", "content": "You are Jankarisetu, a friendly chatbot for a government in India. Use simple Indian-accented English and avoid apostrophes."},
                                        {"role": "user", "content": prompt}
                                    ]
                                    stream = chat_model.chat.completions.create(
                                        messages=messages,
                                        model=selected_model,
                                        temperature=0.1,
                                        top_p=0.1,
                                        stream=True
                                    )
                                    for chunk in stream:
                                        if chunk.choices[0].delta.content is not None:
                                            streamed_text += chunk.choices[0].delta.content
                                            placeholder.markdown(streamed_text, unsafe_allow_html=True)
                                            time.sleep(0.01)
                                st.session_state["chat_history"][-1]["content"] = streamed_text
                                logger.info(f"LLM response: {streamed_text}")
                                # Store in FAQ collection
                                store_faq_query(query_text, streamed_text)
                    except Exception as e:
                        error_msg = f"Error retrieving answer: {str(e)}"
                        logger.error(error_msg)
                        st.session_state["chat_history"].append({"role": "assistant", "content": error_msg})
            else:
                try:
                    prompt = build_conversational_prompt(user_input)
                    st.session_state["chat_history"].append({"role": "assistant", "content": ""})
                    with chat_container:
                        with st.chat_message("assistant"):
                            placeholder = st.empty()
                            streamed_text = ""
                            if selected_model in ["gpt-4o-mini", "gpt-4.1"]:
                                for chunk in chat_model.stream([HumanMessage(content=prompt)]):
                                    streamed_text += chunk.content
                                    placeholder.markdown(streamed_text, unsafe_allow_html=True)
                                    time.sleep(0.01)
                            else:
                                messages = [
                                    {"role": "system", "content": "You are Jankarisetu, a friendly chatbot for a government in India. Use simple Indian-accented English and avoid apostrophes."},
                                    {"role": "user", "content": prompt}
                                ]
                                stream = chat_model.chat.completions.create(
                                    messages=messages,
                                    model=selected_model,
                                    temperature=0.1,
                                    top_p=0.1,
                                    stream=True
                                )
                                for chunk in stream:
                                    if chunk.choices[0].delta.content is not None:
                                        streamed_text += chunk.choices[0].delta.content
                                        placeholder.markdown(streamed_text, unsafe_allow_html=True)
                                        time.sleep(0.01)
                            st.session_state["chat_history"][-1]["content"] = streamed_text
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    logger.error(error_msg)
                    st.session_state["chat_history"].append({"role": "assistant", "content": error_msg})

    st.rerun()