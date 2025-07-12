import os
import streamlit as st
import numpy as np
import pandas as pd
from src.load_file import load_file
from gensim.models import FastText
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.response import generate_response_rag,generate_response_ollama
from src.load_doc_vector import  extract_questions, get_question_vector

from qdrant_client import QdrantClient, models


def faq_prompt(tokenized_sentences, prompt_faq):
    prompt = f"""
        Context: {tokenized_sentences}
        {prompt_faq}
        """
    return prompt

def ques_vari_prompt(tokenized_sentences, question, prompt_faq_variation):
    prompt = f"""
        Original question: {question}
        Context: {tokenized_sentences}
        {prompt_faq_variation}
        """                 
    return prompt


def process_faqs(file_path, prompt_faq, prompt_faq_variation):
    # Initialize questions_df as an empty DataFrame to avoid UnboundLocalError
    questions_df = pd.DataFrame(columns=['Primary Question', 'Variation', 'Similarity Score', 'Answer', 'FileName'])
    
    try:
        # Read content from file
        content = load_file(file_path)
        st.write("File Name: " + file_path)
        content1 = " ".join(content)  # Join all document content

        # Use NLTK's sent_tokenize for more robust sentence splitting
        tokenized_sentences = sent_tokenize(content1)

        # Optionally, preprocess the sentences if needed (e.g., lowercasing)
        tokenized_sentences = [sentence.lower() for sentence in tokenized_sentences]

        # Train FastText model on the tokenized sentences
        fasttext_model = FastText(sentences=tokenized_sentences, vector_size=SentenceTransformer(st.session_state.embed_model).get_sentence_embedding_dimension(), window=5, min_count=1, workers=4)

        #****** Generating Golden Question **********
        st.write(":::: List of Golden Questions ::::")
        ai_response1 = generate_response_ollama(faq_prompt(tokenized_sentences, prompt_faq), st.session_state.temperature)
        ai_response1 = ai_response1.replace('"', "'")
        
        # Extract questions
        questions_list1 = extract_questions(str(ai_response1))
        st.write(questions_list1)

        # ****** Generating similar questions *********
        st.write(":::: List of Similar Questions ::::")
        questions_list2 = []
        questions_list3 = []
        questions_list2 = questions_list1.copy()
        questions_list3 = questions_list1.copy()

        for q in questions_list1:
            questions = []
            questions = generate_response_ollama(ques_vari_prompt(tokenized_sentences, q, prompt_faq_variation), st.session_state.temperature)
            questions = questions.replace('"', "'")
            question = extract_questions(str(questions))
            st.write(question)
            for rq in question:
                questions_list3.append(rq)
                questions_list2.append(q)

        # Create DataFrame
        questions_df = pd.DataFrame({'Primary Question': questions_list3, 'Variation': questions_list2})

        # Convert questions to vectors
        primary_vectors = np.array([get_question_vector(q, fasttext_model) for q in questions_list3])
        variation_vectors = np.array([get_question_vector(q, fasttext_model) for q in questions_list2])

        # Calculate similarity scores
        similarity_scores = cosine_similarity(primary_vectors, variation_vectors)

        # Add similarity scores to DataFrame
        questions_df['Similarity Score'] = [similarity_scores[i].max() for i in range(len(questions_list3))]
        st.write(":::: List of Similar Questions ::::")
        st.write(questions_df)

        #****** Generate Answers for Questions
        questions_df['Answer'] = questions_df['Primary Question'].apply(lambda q: generate_response_rag(q, st.session_state.temperature))
    
        for q in questions_list3:
            questions_df['FileName'] = file_path
  
        st.write(":::: List Answers to question ::::")
        st.write(questions_df)

        # Create the collection directory and subdirectory - faq
        file_path = r'C:\Users\AashishSingh' + st.session_state.vectordb_collection_name
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create directory if it doesn't exist
        file_path = file_path + '/faq'  # Corrected path concatenation
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create directory if it doesn't exist
        faq_path = file_path + "/faq.csv"

        if os.path.exists(faq_path):
            questions_df.to_csv(faq_path, index=False, header=False, mode='a')
        else:
            os.makedirs(os.path.dirname(faq_path), exist_ok=True)  # Use faq_path for directory creation
            questions_df.to_csv(faq_path, index=False, header=True, mode='w')     
    except Exception as e:
        st.error(f"An error occurred: {e}")
        # Return empty DataFrame if an error occurs
        return questions_df

    return questions_df