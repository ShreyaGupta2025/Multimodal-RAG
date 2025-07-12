import streamlit as st
import os
import shutil
import json
import re
import base64
import io
from dotenv import load_dotenv
from src.logo_title import logo_title

# Load environment variables
load_dotenv()

# Validate critical environment variables
required_env_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT_NAME"
]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    st.error(f"Missing environment variables: {', '.join(missing_vars)}. Please add them to the .env file.")
    st.stop()

# Set page configuration to wide layout
st.set_page_config(page_title="Upload & Process PDFs", layout="wide")
logo_title("Upload & Process PDFs")

# Imports for LLM / Summaries / Documents
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI  # Changed to AzureChatOpenAI

# Embeddings
from langchain.embeddings.base import Embeddings
from FlagEmbedding import BGEM3FlagModel

# PDF / Unstructured
from streamlit_pdf_viewer import pdf_viewer
from langchain_community.document_loaders import UnstructuredPDFLoader
import htmltabletomd

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
#from langchain.vectorstores import Qdrant as QdrantVectorStore
from langchain_qdrant import Qdrant as QdrantVectorStore

# NLTK / OCR
import nltk
from unstructured_pytesseract import pytesseract

# LlamaIndex for metadata extraction
import nest_asyncio
nest_asyncio.apply()
from llama_index.llms.azure_openai import AzureOpenAI as LlamaIndexAzureOpenAI  # Changed to AzureOpenAI
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import Document as LlamaDocument
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor, KeywordExtractor, BaseExtractor
from pydantic import Field

# Define BGEM3Embeddings class
class BGEM3Embeddings(Embeddings):
    """
    A wrapper for the BAAI/bge-m3 model from FlagEmbedding.
    Inherits from LangChain's Embeddings so it can be used with QdrantVectorStore.
    """
    def __init__(self, model_name="BAAI/bge-m3", use_fp16=True, batch_size=32, max_length=8192):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.batch_size = batch_size
        self.max_length = max_length

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        outputs = self.model.encode(texts, batch_size=self.batch_size, max_length=self.max_length)
        return outputs["dense_vecs"].tolist()

    def embed_query(self, text: str) -> list[float]:
        outputs = self.model.encode([text], batch_size=1, max_length=self.max_length)
        return outputs["dense_vecs"][0].tolist()

pytesseract.tesseract_cmd = r"tesseract"
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

# ----------------- Global variables & helper functions -----------------
collection_name = "AIS_DOPT"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def image_summarize(img_base64, prompt):
    try:
        chat = AzureChatOpenAI(  # Changed to AzureChatOpenAI
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            temperature=0.1
        )
        msg = chat.invoke([
            HumanMessage(content=(prompt + "\n\nHere is the image in base64:\n" +
                                 f"data:image/jpeg;base64,{img_base64}"))
        ])
        return msg.content
    except Exception as e:
        st.warning(f"Failed to summarize image: {str(e)}")
        return ""

def generate_img_summaries(image_folder):
    img_base64_list = []
    image_summaries = []
    prompt = (
        "You are an assistant tasked with summarizing images for retrieval.\n"
        "Remember these images could potentially contain graphs, charts or tables also.\n"
        "These summaries will be embedded and used to retrieve the raw image for question answering.\n"
        "Give a detailed summary of the image that is well optimized for retrieval.\n"
        "Also give summaries about the image content.\n"
        "Do not add additional words like Summary: etc.\n"
    )
    for fn in sorted(os.listdir(image_folder)):
        if fn.lower().endswith(".jpg"):
            full_path = os.path.join(image_folder, fn)
            try:
                b64_img = encode_image(full_path)
                img_base64_list.append(b64_img)
                summary = image_summarize(b64_img, prompt)
                if summary:
                    image_summaries.append(summary)
                else:
                    image_summaries.append("No summary generated due to processing error.")
            except Exception as e:
                st.warning(f"Error processing image {fn}: {str(e)}")
    return img_base64_list, image_summaries

def load_and_process_pdf(pdf_path, images_path):
    try:
        # Loader for raw text using OCR-only
        loader_ocr = UnstructuredPDFLoader(
            file_path=pdf_path,
            strategy="ocr_only",
            languages=["eng", "hin", "asm"],
            chunking_strategy="by_title",
            combine_text_under_n_chars=300,
            max_characters=1600,
            new_after_n_chars=1600,
            overlap_all=20,
            mode="elements",
        )
        data_text = loader_ocr.load()

        # Loader for tables using hi_res
        loader_hi_res = UnstructuredPDFLoader(
            file_path=pdf_path,
            strategy="hi_res",
            languages=["eng", "hin"],
            infer_table_structure=True,
            chunking_strategy="by_title",
            combine_text_under_n_chars=300,
            max_characters=4000,
            new_after_n_chars=4000,
            overlap_all=20,
            mode="elements",
        )
        data_tables = loader_hi_res.load()

        return {"text": data_text, "tables": data_tables}
    except Exception as e:
        st.error(f"Error loading PDF {pdf_path}: {str(e)}")
        return {"text": [], "tables": []}

def split_docs_and_tables(data):
    docs = data["text"]
    tables = [el for el in data["tables"] if el.metadata.get("text_as_html")]
    for table in tables:
        html_table = table.metadata.get("text_as_html", "")
        try:
            markdown_table = htmltabletomd.convert_table(html_table)
            table.metadata["table_markdown"] = markdown_table
        except Exception as e:
            st.warning(f"Error converting table to markdown: {str(e)}")
            table.metadata["table_markdown"] = html_table
    return docs, tables

def create_summarization_chain(chat_model):
    prompt_text = (
        "You are an assistant tasked with summarizing tables and text particularly for semantic retrieval.\n"
        "These summaries will be embedded and used to retrieve the raw text or table elements.\n"
        "Detect the language of the provided text. If the text is in Hindi, provide the summary in Hindi; "
        "if the text is in English, provide the summary in English. If the text is in Assamese provide the summary in Assamese\n"
        "Give a detailed summary of the table or text below that is well optimized for retrieval.\n"
        "For any tables also add a one-line description of what the table is about besides the summary.\n"
        "Do not add additional words like 'Summary:' etc.\n\n"
        "Table or text chunk:\n"
        "{element}"
    )
    prompt = ChatPromptTemplate.from_template(prompt_text)
    summarize_chain = (
        {"element": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )
    return summarize_chain

def summarize_chunks(docs, tables, summarize_chain):
    text_chunks = [doc.page_content for doc in docs]
    table_chunks = [tbl.metadata.get("table_markdown", "") for tbl in tables]
    try:
        text_summaries = summarize_chain.batch(text_chunks, {"max_concurrency": 5})
        table_summaries = summarize_chain.batch(table_chunks, {"max_concurrency": 5})
    except Exception as e:
        st.error(f"Error summarizing chunks: {str(e)}")
        return [], []
    return text_summaries, table_summaries

class MyCustomExtractor(BaseExtractor):
    title_extractor: TitleExtractor = Field(..., description="Title extractor")
    questions_extractor: QuestionsAnsweredExtractor = Field(..., description="Questions extractor")
    keyword_extractor: KeywordExtractor = Field(..., description="Keyword extractor")
    def extract(self, nodes):
        try:
            titles = self.title_extractor.extract(nodes)
            questions = self.questions_extractor.extract(nodes)
            keywords = self.keyword_extractor.extract(nodes)
            merged_metadata = []
            for i, node in enumerate(nodes):
                new_meta = {}
                new_meta.update(titles[i])
                new_meta.update(questions[i])
                new_meta.update(keywords[i])
                merged_metadata.append(new_meta)
            return merged_metadata
        
        except Exception as e:
            st.warning(f"Error in metadata extraction: {str(e)}")
            return [{} for _ in nodes]
        
    async def aextract(self, nodes):
        return self.extract(nodes)

def extract_metadata_with_llamaindex(content_str, file_level_title, file_name):
    try:
        doc = LlamaDocument(text=content_str)
        text_splitter = TokenTextSplitter(separator=" ", chunk_size=1000, chunk_overlap=128)
        llm_for_extractors = LlamaIndexAzureOpenAI(  # Changed to LlamaIndexAzureOpenAI
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            temperature=0.1
        )
        title_ext = TitleExtractor(nodes=1, llm=llm_for_extractors)
        qa_ext = QuestionsAnsweredExtractor(questions=3, llm=llm_for_extractors)
        kw_ext = KeywordExtractor(keywords=5, llm=llm_for_extractors)
        my_extractor = MyCustomExtractor(
            title_extractor=title_ext,
            questions_extractor=qa_ext,
            keyword_extractor=kw_ext
        )
        pipeline = IngestionPipeline(transformations=[text_splitter, my_extractor])
        nodes = pipeline.run(documents=[doc])
        for node in nodes:
            node.metadata["file_level_title"] = file_level_title
            node.metadata["file_name"] = file_name
        return nodes
    except Exception as e:
        st.error(f"Error extracting metadata with LlamaIndex: {str(e)}")
        return []

def ensure_folder(folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)

def save_processed_data_to_json(save_folder: str, pdf_name: str,
                               text_chunks_info: list,
                               table_chunks_info: list,
                               image_chunks_info: list,
                               embeddings):
    try:
        output_path = os.path.join(save_folder, "processed_data.json")
        # Compute and store embeddings for each chunk
        for chunk in text_chunks_info:
            chunk["embedding"] = embeddings.embed_query(chunk["summary"])
        for chunk in table_chunks_info:
            chunk["embedding"] = embeddings.embed_query(chunk["summary"])
        for chunk in image_chunks_info:
            chunk["embedding"] = embeddings.embed_query(chunk["summary"])
        data_dict = {
            "pdf_name": pdf_name,
            "text_chunks": text_chunks_info,
            "table_chunks": table_chunks_info,
            "image_chunks": image_chunks_info,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error saving processed data to JSON: {str(e)}")

def store_summaries_in_qdrant(text_summaries, table_summaries,
                              docs, tables,
                              img_base64_list, image_summaries,
                              embeddings, batch_size=100):
    try:
        emb_dim = 1024  # BGE-M3 embedding dimension
        client = QdrantClient(url="http://localhost:6333", timeout=10)
        
        # Test Qdrant connectivity
        try:
            client.get_collections()
        except Exception as e:
            st.error(f"Failed to connect to Qdrant server: {str(e)}")
            return None, []

        # Check if collection exists
        collections = client.get_collections().collections
        exists = any(collection.name == collection_name for collection in collections)

        # Create collection if it doesn't exist
        if not exists:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=emb_dim, distance=Distance.COSINE),
            )
            st.info(f"Created new collection: {collection_name}")
        else:
            st.info(f"Using existing collection: {collection_name}")

        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings
        )
        
        summary_docs = []

        # Process text chunks
        for idx, summ in enumerate(text_summaries):
            orig_chunk = docs[idx]
            chunk_str = orig_chunk.page_content
            file_name = orig_chunk.metadata.get("filename", "unknown_file.pdf")
            file_title = orig_chunk.metadata.get("title", "Generic Title")
            nodes = extract_metadata_with_llamaindex(chunk_str, file_title, file_name)
            node_meta = nodes[0].metadata if nodes else {}

            for k, v in orig_chunk.metadata.items():
                node_meta.setdefault(k, v)

            node_meta["raw_text"] = chunk_str
            combined_content = f"Raw Text:\n{chunk_str}\n\n---\n\nSummary:\n{summ}"
            doc_for_store = Document(page_content=combined_content, metadata=node_meta)
            summary_docs.append(doc_for_store)
        
        # Process tables
        for idx, summ in enumerate(table_summaries):
            orig_table = tables[idx]
            chunk_str = orig_table.metadata.get("table_markdown", "")
            file_name = orig_table.metadata.get("filename", "unknown_file.pdf")
            file_title = orig_table.metadata.get("title", "Generic Title")
            nodes = extract_metadata_with_llamaindex(chunk_str, file_title, file_name)
            node_meta = nodes[0].metadata if nodes else {}

            for k, v in orig_table.metadata.items():
                node_meta.setdefault(k, v)
            node_meta["raw_text"] = chunk_str
            
            combined_content = f"Raw Table Markdown:\n{chunk_str}\n\n---\n\nSummary:\n{summ}"
            
            doc_for_store = Document(page_content=combined_content, metadata=node_meta)
            summary_docs.append(doc_for_store)
        
        # Process images
        image_docs = []

        for i, summary_text in enumerate(image_summaries):
            b64_img = img_base64_list[i]
            nodes = extract_metadata_with_llamaindex(
                summary_text,
                file_level_title="Image Chunk",
                file_name=f"image_{i}.jpg"
            )
        
            meta_img = nodes[0].metadata if nodes else {}
            meta_img["is_image"] = True
            meta_img["image_base64"] = b64_img
        
            combined_content = f"Image Summary:\n{summary_text}"
            doc_for_store = Document(page_content=combined_content, metadata=meta_img)
            image_docs.append(doc_for_store)
        
        all_docs = summary_docs + image_docs
        
        # Add documents in batches
        st.write(f"Processing {len(all_docs)} documents for embedding and storage...")
        for i in range(0, len(all_docs), batch_size):
            batch = all_docs[i:i + batch_size]
            vectorstore.add_documents(batch)
            st.write(f"Processed batch {i//batch_size + 1} of {len(all_docs)//batch_size + 1}")
        
        st.success(f"Added {len(all_docs)} documents to the collection")
    
        return vectorstore, all_docs
    
    except Exception as e:
        st.error(f"Error storing summaries in Qdrant: {str(e)}")
        return None, []

def upload_page():

    st.title("Upload & Process PDFs")
    st.write("Please upload one or more PDF files to process and store the content.")
    st.info("After selecting PDF files, click the 'Process PDFs' button.")

    # Initialize embeddings
    embeddings = None

    try:
        st.write("Initializing embeddings...")
        embeddings = BGEM3Embeddings(
            model_name="BAAI/bge-m3",
            use_fp16=True,
            batch_size=32,
            max_length=8192
        )
        st.write("Embeddings initialized successfully.")

    except Exception as e:
        st.error(f"Failed to initialize embeddings: {str(e)}")
        st.write("You can still view the UI, but PDF processing will not work until embeddings are fixed.")
        embeddings = None

    # Initialize chat model
    chat_model = None
    try:
        st.write("Initializing chat model...")
        chat_model = AzureChatOpenAI(  # Changed to AzureChatOpenAI
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            temperature=0
        )
        st.write("Chat model initialized successfully.")
    except Exception as e:
        st.error(f"Failed to initialize chat model: {str(e)}")
        st.write("You can still view the UI, but summarization will not work until the chat model is fixed.")
        chat_model = None

    # Display file uploader
    st.write("Starting file uploader...")
    uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

    if not uploaded_files:
        st.warning("No files uploaded yet. Please upload PDF files to proceed.")
        return

    if st.button("Process PDFs"):
        if not embeddings:
            st.error("Cannot process PDFs: Embeddings initialization failed. Please check the error above.")
            return
        if not chat_model:
            st.error("Cannot process PDFs: Chat model initialization failed. Please check the error above.")
            return

        st.write("Creating summarization chain...")
        summarize_chain = create_summarization_chain(chat_model)
        st.write("Summarization chain created successfully.")

        for file_idx, uploaded_file in enumerate(uploaded_files):
            safe_pdf_name = sanitize_filename(uploaded_file.name)
            pdf_folder = f"{safe_pdf_name}_files"
            ensure_folder(pdf_folder)
            pdf_path = os.path.join(pdf_folder, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.subheader(f"Processing PDF #{file_idx + 1}: {uploaded_file.name}")
            st.write(f"Saved PDF locally at: {pdf_path}")
            with st.expander(f"1) View PDF ({uploaded_file.name})"):
                pdf_viewer(uploaded_file.getvalue(), width=700)
            images_path = "./figures"
            if os.path.exists(images_path):
                shutil.rmtree(images_path)
            os.makedirs(images_path)
            with st.expander(f"2) Parse {uploaded_file.name}"):
                st.write("Parsing PDF...")
                data = load_and_process_pdf(pdf_path, images_path)
                st.write(f"Number of text elements loaded: {len(data['text'])}")
                valid_tables = [el for el in data["tables"] if el.metadata.get("text_as_html")]
                st.write(f"Number of table elements loaded: {len(valid_tables)}")
            combined_pages = {}
            for el in data["text"]:
                page = el.metadata.get("page")
                if page is None:
                    page = f"index_{data['text'].index(el)}"
                combined_pages.setdefault(page, []).append(("text", el.page_content))
            for el in data["tables"]:
                if not el.metadata.get("text_as_html"):
                    continue
                page = el.metadata.get("page")
                if page is None:
                    page = f"index_{data['tables'].index(el)}"
                table_html = el.metadata.get("text_as_html", "")
                if table_html:
                    try:
                        table_markdown = htmltabletomd.convert_table(table_html)
                        combined_pages.setdefault(page, []).append(("table", table_markdown))
                    except Exception as e:
                        st.error(f"Error converting table to markdown: {str(e)}")
                        combined_pages.setdefault(page, []).append(("table", table_html))
            try:
                sorted_pages = sorted(combined_pages.keys(), key=lambda x: int(x))
            except Exception:
                sorted_pages = sorted(combined_pages.keys())
            page_texts = []
            for page in sorted_pages:
                items = combined_pages[page]
                page_content = "\n\n".join(item[1] for item in items)
                page_texts.append(f"Page {page}:\n\n{page_content}")
            original_text = "\n\n---\n\n".join(page_texts)
            md_file_path = os.path.join(pdf_folder, "original_text.md")
            with open(md_file_path, "w", encoding="utf-8") as md_file:
                md_file.write(original_text)
            with st.expander("View Original Text Markdown"):
                with open(md_file_path, "r", encoding="utf-8") as md_file:
                    md_content = md_file.read()
                st.markdown(md_content, unsafe_allow_html=False)
            with st.expander("3) Split into Text & Table chunks"):
                st.write("Splitting documents and tables...")
                docs, tables = split_docs_and_tables(data)
                st.write(f"Found {len(docs)} text chunks, {len(tables)} table chunks.")
            with st.expander("4) Summarize Text & Tables"):
                st.write("Summarizing chunks...")
                text_summaries, table_summaries = summarize_chunks(docs, tables, summarize_chain)
                st.write(f"Summaries => {len(text_summaries)} text / {len(table_summaries)} tables")
            with st.expander("5) Extract & Summarize Images"):
                st.write("Processing images...")
                if os.path.exists(images_path):
                    imgs_base64, image_summaries = generate_img_summaries(images_path)
                    st.write(f"Extracted {len(imgs_base64)} images.")
                else:
                    imgs_base64, image_summaries = [], []
                    st.write("No images folder found.")
            text_chunks_info = []
            if text_summaries:
                with st.expander("Text Chunks"):
                    for i, doc_obj in enumerate(docs):
                        chunk_item = {
                            "chunk_index": i + 1,
                            "chunk_text": doc_obj.page_content,
                            "metadata": doc_obj.metadata,
                            "summary": text_summaries[i],
                        }
                        text_chunks_info.append(chunk_item)
                        st.markdown(f"**Text Chunk {i + 1}**")
                        st.write("**Original Text:**", doc_obj.page_content)
                        st.write("**Metadata:**", doc_obj.metadata)
                        st.write("**Summary:**", text_summaries[i])
                        st.markdown("---")
            table_chunks_info = []
            if table_summaries:
                with st.expander("Table Chunks"):
                    for i, tbl_obj in enumerate(tables):
                        md_table = tbl_obj.metadata.get("table_markdown", "")
                        chunk_item = {
                            "table_index": i + 1,
                            "markdown_table": md_table,
                            "metadata": tbl_obj.metadata,
                            "summary": table_summaries[i],
                        }
                        table_chunks_info.append(chunk_item)
                        st.markdown(f"**Table {i + 1}**")
                        st.markdown(md_table)
                        st.write("**Metadata:**", tbl_obj.metadata)
                        st.write("**Summary:**", table_summaries[i])
                        st.markdown("---")
            image_chunks_info = []
            if imgs_base64:
                with st.expander("Images"):
                    for i, (b64_img, summary_text) in enumerate(zip(imgs_base64, image_summaries)):
                        chunk_item = {"image_index": i + 1, "image_base64": b64_img, "summary": summary_text}
                        image_chunks_info.append(chunk_item)
                        decoded_image = base64.b64decode(b64_img)
                        st.markdown(f"**Image {i + 1}**")
                        st.image(io.BytesIO(decoded_image), caption=f"Image {i + 1}")
                        st.write("**Summary:**", summary_text)
                        st.markdown("---")
            st.write("Saving processed data to JSON...")
            save_processed_data_to_json(
                save_folder=pdf_folder,
                pdf_name=uploaded_file.name,
                text_chunks_info=text_chunks_info,
                table_chunks_info=table_chunks_info,
                image_chunks_info=image_chunks_info,
                embeddings=embeddings
            )
            st.write("JSON data saved successfully.")
            # Store the summaries in Qdrant for this PDF
            st.subheader(f"Storing Data in Qdrant for {uploaded_file.name}")
            vectorstore, pdf_docs = store_summaries_in_qdrant(
                text_summaries,
                table_summaries,
                docs,
                tables,
                imgs_base64,
                image_summaries,
                embeddings
            )
            if vectorstore:
                st.session_state[f"docs_{safe_pdf_name}"] = pdf_docs
                st.write(f"Stored {len(pdf_docs)} documents for {uploaded_file.name}")
                st.success(f"Processing and storage complete for {uploaded_file.name}!")
            else:
                st.error(f"Failed to store documents for {uploaded_file.name} in Qdrant.")
            st.markdown("---")

        st.success("All PDFs processed and stored successfully!")

if __name__ == "__main__":
    st.write("Starting Streamlit application...")
    upload_page()