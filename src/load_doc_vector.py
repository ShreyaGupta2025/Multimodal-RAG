import re
import string
import streamlit as st
import uuid
import json

import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from gensim.models import FastText
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from qdrant_client.http import models as rest

from sentence_transformers import SentenceTransformer

from src.load_file import load_file
from src.intent_classifier import intent_topic_modeling, process_chunk_data

# Download NLTK resources
try:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    stop_words = set(stopwords.words('english'))
except Exception as e:
    st.error(f"An error occurred: {e}")


def clean_text(text):
    # Remove unwanted characters using regex
    cleaned_text = re.sub(r'[#,?.,!;]', '', text)  # You can add more characters to this list if needed
    cleaned_text = cleaned_text.lower()  # Convert to lowercase if necessary
    return cleaned_text


def extract_questions(text):
    #Extract numbered questions from text.
    try:
        pattern = r'(\d+)\.\s(.+?)(?=\d+\.|$)'
        cleaned_text = re.sub(r'(\d+)\.\s(.+?)\s*(".*|additional_kwargs.*|response_metadata.*)',
                            r'\1. \2', text, flags=re.DOTALL)
        matches = re.findall(pattern, cleaned_text, re.DOTALL)
        questions = [match[1].replace('\\n', '').replace("\\", "").replace("'", "").strip() for match in matches]
        #questions = [match[1].replace('\\n', '').replace("\\", "").replace('"',"'").strip() for match in matches]
    except Exception as e:
        st.error(f"An error occurred: {e}")
    return questions


def upload_document(file_path: str, collection_name, qdrant_client, embed_model, chunk_size, chunk_overlap, filename):
  
    documents = load_file(file_path)
    st.text_area("", documents, height=300)
    document_text = " ".join(documents)  # Join all document content
  
    # clean document_text to remove stopwords and symbols and stemmization
    document_text = clean_text(document_text)
    st.text_area("Clean Text", document_text, height=300)

    #document_test = extract_questions(document_text)
    #st.text_area("Clean2 Text", document_text, height=300)

    # Use NLTK's sent_tokenize for more robust sentence splitting
    tokenized_sentences = sent_tokenize(document_text)
    
    # Optionally, preprocess the sentences if needed (e.g., lowercasing)
    tokenized_sentences = [sentence.lower() for sentence in tokenized_sentences]
    st.text_area("Tokenized sentences", tokenized_sentences, height=300)

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
    chunks = splitter.create_documents(tokenized_sentences)
    st.text_area("Chunks", chunks, height=300)

    for idx, chunk in enumerate(chunks):
        chunk.metadata = {
            "document_name": filename,
            "services": ["citizen"],
            "intent": process_chunk_data(chunk) #intent_topic_modeling(chunk)
        }

        st.write("Chunk# ",idx, "Content:", chunk.page_content, "\nEmbedding:",json.dumps(SentenceTransformer(embed_model).encode(chunk.page_content).tolist()), "\n\nMetadata:", chunk.metadata)

        try:
            qdrant_client.upload_points(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()), #id=idx,
                        vector=SentenceTransformer(embed_model).encode(chunk.page_content).tolist(),
                        payload={"content": chunk.page_content, "metadata": chunk.metadata} 
                    ) 
                ]
            )
            st.write(f"Successfully loaded the chunks of file {file_path} into vector db name")
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            return

def preprocess_text(text):
    # Preprocess the input text: lowercasing, removing stop words and punctuation.
    try:
        text = text.lower()
        tokens = nltk.sent_tokenize(text)
        tokens = [word for word in tokens if word not in string.punctuation and word not in stop_words]
    except Exception as e:
        st.error(f"An error occurred: {e}")
    return tokens

def get_embedding(text, embed_model):
    # Create embeddings for your data (replace with actual embeddings)
    try:
        # Tokenize the content to train FastText
        tokenized_sentence = sent_tokenize(text.lower())

        # Train FastText model on the tokenized sentences
        fasttext_model = FastText(sentences=tokenized_sentence, vector_size=SentenceTransformer(embed_model).get_sentence_embedding_dimension(), window=5, min_count=1, workers=4)

        tokens = preprocess_text(text)
        vectors = [fasttext_model.wv[word] for word in tokens if word in fasttext_model.wv]
        
        # Replace this with your actual embedding logic (e.g., using a model)
        #return np.random.rand(300).tolist()  # Example random embeddings
    except Exception as e:
        st.error(f"An error occurred: {e}")

    return np.mean(vectors, axis=0) if vectors else np.zeros(fasttext_model.vector_size) 

def get_question_vector(question, embed_model):
    # Convert a question to a vector by averaging the word vectors of its words using FastText.
    try:
        tokens = preprocess_text(question)
        vectors = [embed_model.wv[word] for word in tokens if word in embed_model.wv]
    except Exception as e:
        st.error(f"An error occurred: {e}")
    return np.mean(vectors, axis=0) if vectors else np.zeros(embed_model.vector_size) 

def upload_faq_vector(df,q, qdrant_client, embed_model, collection_name_faq):
    # Uploaded the questions, variation, similarity score, answers and filename in vector database
    try:
        # Prepare points to upsert
        points_to_upsert = []
        count_result = qdrant_client.count(collection_name_faq)
        total_points=count_result.count
    
        for index, row in df.iterrows():
            embedding = get_embedding(row['Primary Question'],  SentenceTransformer(embed_model))
            points_to_upsert.append(
                rest.PointStruct(
                    id=index+total_points,
                    vector=embedding,
                    payload={"question": row['Primary Question'], "response": row['Answer'], "citation":q}
                    #payload={"question": row['Primary Question'], "response": row['Variation']}  
                )
            )
    except Exception as e:
        st.error(f"An error occurred: {e}")        
    
    # Step 4: Upload points to Qdrant
    try:
        qdrant_client.upsert(collection_name=collection_name_faq, points=points_to_upsert)
    except Exception as e:
        st.error(f"Error during upsert: {str(e)}")
    
    # Confirming upload
    st.write(f"Uploaded {len(points_to_upsert)} points to Qdrant.")