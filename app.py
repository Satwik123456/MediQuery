import os
import pickle
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Paths
INDEX_DIR = "./index"
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
MAPPING_PATH = os.path.join(INDEX_DIR, "mapping.pkl")

# Load FAISS index
index = faiss.read_index(FAISS_INDEX_PATH)

# Load mapping
with open(MAPPING_PATH, "rb") as f:
    mapping = pickle.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# OpenAI client
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()
print(os.getenv("OPENAI_API_KEY"))

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Streamlit app
st.set_page_config(page_title="Medical FAQ Chatbot", page_icon="💊")
st.title("💊 Medical FAQ Chatbot")
st.write("Ask me any medical FAQ (Disclaimer: Not a substitute for professional medical advice).")

query = st.text_input("Enter your question:")

if query:
    # Embed query
    q_embedding = model.encode([query])
    q_embedding = np.array(q_embedding).astype("float32")

    # Search in FAISS
    k = 3
    D, I = index.search(q_embedding, k)

    # Retrieve top answers
    retrieved_context = [mapping[idx]["question"] + " - " + mapping[idx]["answer"] for idx in I[0]]

    # Join context
    context_text = "\n".join(retrieved_context)

    # Send to OpenAI LLM
    prompt = f"""You are a helpful medical assistant.
Use the context below to answer the user’s question.
If the context is not relevant, politely say you don’t know.

Context:
{context_text}

Question:
{query}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a medical assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
    )

    st.subheader("Answer:")
    st.write(response.choices[0].message.content)
