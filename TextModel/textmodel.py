import os
import json
import re
import fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

def is_pdf_query(query):
    return query.lower().startswith("database")

def clean_text(input_text):
    cleaned = re.sub(r"[^a-zA-Z0-9\s,.!?'-]", "", input_text)
    return cleaned

def extract_text_from_pdfs(pdf_folder):
    pdf_texts = []
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith('.pdf'):
            with fitz.open(os.path.join(pdf_folder, file_name)) as pdf_doc:
                text = ""
                for page in pdf_doc:
                    text += page.get_text()
                pdf_texts.append(text)
    return pdf_texts

def embed_texts(texts, model):
    embeddings = model.encode(texts)
    return embeddings

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

client = Groq()
pdf_folder = os.path.join(working_dir, 'dataset_location')  
model = SentenceTransformer('all-MiniLM-L6-v2')  
pdf_texts = extract_text_from_pdfs(pdf_folder)
pdf_embeddings = embed_texts(pdf_texts, model)
index = create_faiss_index(np.array(pdf_embeddings))
chat_history = []
print("AI Launching")
print("AI Launched ........\n")

while True:
    user_prompt = input("You: ").strip()
    if user_prompt.lower() == "exit":
        print("Exiting the chat...")
        break

    cleaned_user_prompt = clean_text(user_prompt)
    if is_pdf_query(cleaned_user_prompt):
        query_embedding = model.encode([cleaned_user_prompt])
        D, I = index.search(np.array(query_embedding), k=1)  
        matched_text = pdf_texts[I[0][0]]
        chat_history.append({"role": "user", "content": cleaned_user_prompt})
        recent_history = chat_history[0:]
        messages = [
            {"role": "system", "content": "You are an AI invented by Kavin   Every query you answer should be concise and under 50 words."},
            *recent_history,
            {"role": "user", "content": f"Based on the database content: {matched_text[:1000]}..."}
        ]
        try:
            response = client.chat.completions.create(
                model="llama-3.1-70b-versatile",  
                messages=messages
            )
            assistant_response = response.choices[0].message.content.strip()
            assistant_response = assistant_response.replace("PDF", "database")
        except Exception as e:
            assistant_response = f"An error occurred: {e}"
    else:
        messages = [
            {"role": "system", "content": "Your an AI invented by Kavin. You are a helpful and knowledgeable assistant. Every answer should be concise and under 50 words."},
            {"role": "user", "content": cleaned_user_prompt}
        ]
        try:
            response = client.chat.completions.create(
                model="llama-3.1-70b-versatile", 
                messages=messages
            )
            assistant_response = response.choices[0].message.content.strip()
            assistant_response = assistant_response.replace("PDF", "database")
        except Exception as e:
            assistant_response = f"An error occurred: {e}"
    
    print(f"AI: {assistant_response}")
