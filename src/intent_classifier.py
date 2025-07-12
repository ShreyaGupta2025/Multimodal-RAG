import os
import re
import streamlit as st
import openai
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np
from bertopic import BERTopic
from transformers import AutoTokenizer, AutoModel
import torch

"""
Approach 1. Rule-based Approach (Keyword Matching)

Strategies to Programmatically Determine Metadata

Here are some approaches to programmatically determine metadata (e.g., intent and purpose) for each chunk:

    Rule-based Extraction: For well-structured data (e.g., legal documents or reports), you can use predefined rules or keyword matching to identify metadata like laws, intents, or topics.

    Pre-trained NLP Models: You can use NLP models to classify or tag chunks with labels such as intent, topic, purpose, or other categories. Models like BERT, GPT, or specific task-trained models can help extract such information.

    Topic Modeling: Use unsupervised learning techniques like Latent Dirichlet Allocation (LDA) or more recent methods like BERTopic to automatically determine topics or categories for each chunk.

    Sentence Classification or Text Classification: You can train a supervised model to classify text chunks into categories (such as intent, legal regulation, informational, etc.).

    Custom NLP Pipelines: Combine a few NLP tools like Named Entity Recognition (NER), text summarization, or sentiment analysis to infer metadata for each chunk.

    Explanation:

    The function extract_intent_from_chunk() checks for specific keywords (e.g., "privacy", "user consent", "law") in the text and returns an associated intent.
    This is a simple, rule-based approach that works well for structured documents where the intent is keyword-driven.
    
    """

def intent_rule_based(chunk):
    chunk = [chunk] if isinstance(chunk, str) else chunk

    if re.search(r'\b(protect|privacy|confidential)\b', chunk, re.IGNORECASE):
        return 'Privacy Protection'
    elif re.search(r'\b(user|consent|rights)\b', chunk, re.IGNORECASE):
        return 'User Consent'
    elif re.search(r'\b(regulation|law|act)\b', chunk, re.IGNORECASE):
        return 'Legal Regulation'
    else:
        return 'General Information'


"""
Approach 2. Using Pre-trained NLP Models (e.g., BERT or GPT)

For more dynamic and nuanced intent detection, you can use a pre-trained model like BERT, GPT, 
or a specialized text classifier. Hugging Face's transformers library provides many 
pre-trained models for tasks like text classification, question answering, or sentiment analysis.

Explanation:

    We use a zero-shot classification pipeline from Hugging Face, which allows us to classify text into predefined labels (like 'Privacy Protection', 'Legal Regulation', etc.).
    The model (facebook/bart-large-mnli) can classify the chunk of text into one of the intents without the need for training a custom model.
"""
def process_chunk_data(chunk):
    # Extract the 'page_content' as the document text
    if hasattr(chunk, 'page_content'):
        document_text = chunk.page_content
    else:
        raise ValueError("Chunk does not contain 'page_content'.")
    
    # Pass the extracted text (document) to intent_nlp
    return intent_nlp([document_text])  # Make sure it's passed as a list of strings


def intent_nlp(documents):
    # Initialize the classifier
    classifier = pipeline("zero-shot-classification",model="facebook/bart-large-xnli")

    candidate_labels = ["information retrieval", "answering questions", "document categorization"]  # Example labels

    # Ensure documents is a list of strings
    if isinstance(documents, str):
        documents = [documents]  # Convert to list if it's a single string

    if not all(isinstance(doc, str) for doc in documents):
        raise ValueError("Each document should be a string.")

    # Ensure candidate labels are not empty
    if len(candidate_labels) == 0:
        raise ValueError("Candidate labels must not be empty.")

    # Check if documents are empty
    if len(documents) == 0:
        raise ValueError("Documents are empty!")

    # Process the documents
    result = classifier(documents, candidate_labels)
    
    return result


"""
Approach 3: Topic Modeling (Unsupervised Approach)

If the intent or purpose is more ambiguous, you can use topic modeling techniques 
to automatically discover themes or topics in the text. Methods like Latent Dirichlet 
Allocation (LDA) or BERTopic (which uses transformers) are great choices.

Explanation:

    BERTopic uses transformer-based embeddings to identify topics in text.
    It returns topics and the top words that describe each topic.
    You can use this to extract the primary topic or purpose of each chunk. 
    Once topics are identified, you can map them to intent or purpose manually or automatically.

"""
# Initialize tokenizer and model for embeddings (you can replace this with your own embedding model)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Function to generate embeddings for documents using the transformer model
def get_embeddings(documents):
    # Tokenize the documents
    inputs = tokenizer(documents, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        # Get embeddings from the transformer model
        outputs = model(**inputs)
    
    # Use mean of the last hidden state as the document embedding (you can experiment with pooling strategies)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Average across all tokens
    
    # Convert PyTorch tensor to NumPy array (required by BERTopic)
    embeddings = embeddings.cpu().numpy()
    return embeddings

# Function to process the chunk and get topics using BERTopic
def intent_topic_modeling(chunk):
    # Extract the page content from the chunk (assuming 'chunk' is a Document object with 'page_content')
    document_text = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
    
    # Ensure document_text is a string
    document_text = str(document_text)
    
    # Create a list of documents (in this case, just one document)
    documents = [document_text]
    
    # Print for debugging: inspect the first 200 characters of the document
    print(f"Document: {documents[0][:200]}...")

    # Generate embeddings for the documents using the custom transformer model
    embeddings = get_embeddings(documents)
    
    # Print the shape of the embeddings (ensure it's valid)
    print(f"Embeddings shape: {embeddings.shape}")

    # Check if embeddings are empty or invalid (e.g., NaN)
    if embeddings is None or embeddings.shape[0] == 0 or np.isnan(embeddings).any():
        raise ValueError("Embeddings are empty or invalid!")
    
    # Initialize BERTopic
    topic_model = BERTopic()

    # Fit the BERTopic model and assign topics to documents
    topics, probabilities = topic_model.fit_transform(documents, embeddings)

    # Return the topics
    return topics


class Document:
    def __init__(self, page_content):
        self.page_content = page_content
        self.metadata = {}


"""
Approach 4. Named Entity Recognition (NER) and Other NLP Tools

Another approach is to use Named Entity Recognition (NER) to identify specific 
entities like laws, acts, or regulations mentioned in the text. You can then link 
these to specific metadata fields.

Explanation:

    Hugging Face's NER pipeline can automatically extract named entities from text, such as laws, 
    regulations, dates, and other important terms.
    This can help populate metadata fields like laws or topics when processing chunks.

"""
# Set device to GPU if available, else CPU
device = 0 if torch.cuda.is_available() else -1

# Define the NER function using Hugging Face's transformers library
def intent_name_entity_rel(chunk):
    # Load a pre-trained NER model from Hugging Face (use a model like BERT for NER)
    nlp_ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", device=device)

    # If 'chunk' is not a list, convert it into one
    chunk = [chunk] if isinstance(chunk, str) else chunk

    # Process the chunk with Hugging Face's NER pipeline
    entities = []
    for text in chunk:
        # Run NER on the text
        ner_results = nlp_ner(text)
        entities.append([(ent['word'], ent['entity']) for ent in ner_results])

    print("Entities found:", entities)
    return entities

# Function to process chunk data (example usage with NER)
def process_chunk_data(chunk):
    # Extract the 'page_content' as the document text
    if hasattr(chunk, 'page_content'):
        document_text = chunk.page_content
    else:
        raise ValueError("Chunk does not contain 'page_content'.")
    
    # Pass the extracted text (document) to the NER function
    return intent_name_entity_rel([document_text])  # Make sure it's passed as a list of strings

# Example usage with a sample chunk
sample_chunk = "This section discusses the General Data Protection Regulation (GDPR)."
entities = intent_name_entity_rel(sample_chunk)
print(entities)

"""
Approach 5: 5. Using a Pre-trained Intent Detection Model (e.g., GPT-3)

If you have complex or varied intents and you need to detect very specific 
intents or actions, a model like GPT-3 or other generative models can be helpful. 
With GPT-3, you can prompt the model to extract intent based on the text content.

Explanation:

    GPT-3 can generate human-like responses based on prompts.
    By providing a chunk and asking it to identify intent, you can leverage GPT-3's 
    powerful language understanding to detect the purpose of a chunk.

"""
def intent_gpt(chunk):

    # Define your OpenAI API key
    openai.api_key = 'your-api-key'

    # Example chunk
    chunk = [chunk] if isinstance(chunk, str) else chunk
    #chunk = "This section discusses the user's right to withdraw consent under GDPR."

    # Use GPT-3 to determine intent
    response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=f"Identify the intent of the following text: {chunk}",
    max_tokens=60
    )

    # Extract the result
    intent = response.choices[0].text.strip()
    print(f"Intent: {intent}")
    return intent
