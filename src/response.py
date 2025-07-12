import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
from langchain_openai import AzureChatOpenAI
from langchain_ollama import OllamaLLM  # Kept for commented-out code
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def search_response_chunk(query: str, collection_name_doc, qdrant_client, embed_model):
    # https://github.com/alfredodeza/learn-retrieval-augmented-generation/blob/main/examples/2-embeddings/embeddings.ipynb
    # Search time for awesome wines!
    results = qdrant_client.search(
        collection_name=collection_name_doc,
        query_vector=SentenceTransformer(embed_model).encode(query).tolist(),
        limit=5
    )
    return results

def generate_response_ollama(prompt1: str, temp):
    # Active implementation using Azure OpenAI
    try:
        # Initialize Azure OpenAI client
        llm = AzureChatOpenAI(
            model="gpt-4o-mini",
            temperature=temp,
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2025-01-01-preview"
        )
        response = llm.invoke(prompt1)
        return response.content  # Extract the content from the response
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []

    # Commented-out implementation using Ollama
    """
    # Generate response using LLM model deployed on the laptop
    try:
        llm = OllamaLLM(model="llama3", temperature=temp)
        response = llm.invoke(prompt1)
        return response
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []
    """

def generate_response_rag(prompt, temperature):   
    summary = generate_response_ollama(str(prompt), temperature)
    return summary

def generate_response_anythingllm(question):
    # Anything LLM-Define the API endpoint and headers
    api_url = 'http://localhost:3001/api/v1/workspace/sampleup/chat'  # move to configuration file
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer 7T1Z8A2-MBG4EHQ-KJTKXE6-JJ257H6",  # move to configuration file
        "Content-Type": "application/json"
    }  
    # Prepare the payload for the API call
    # st.text(question)
    payload = {
        "message": f"{question}",
        "mode": "chat"
    }
    
    # Make the API call to generate the questions answer
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        # st.write(response)
    except Exception as e:
        st.error(f"An error occurred: {e}")

    if response.status_code == 200:
        # Parse and display the generated lesson plan
        answer = response.json().get("textResponse", "No answer generated.")
    else:
        st.error("Failed to generate the answer of the given question. Please try again.")
    return answer