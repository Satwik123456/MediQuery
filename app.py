# app.py
import os
import pickle
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Config
INDEX_DIR = "./index"
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
MAPPING_PATH = os.path.join(INDEX_DIR, "mapping.pkl")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # for embeddings
QA_MODEL_NAME = "distilbert-base-cased-distilled-squad"  # extractive QA model

st.set_page_config(page_title="MediQuery (Local RAG)", layout="centered")
st.title("MediQuery — Local RAG Medical FAQ Chatbot")
st.markdown("**Note:** Local-only, free — answers are extractive from the FAQ knowledge base.")

# Check index files
if not os.path.exists(FAISS_PATH) or not os.path.exists(MAPPING_PATH):
    st.error("Index not found. Run `python build_index.py --input medical_faqs_100.csv --index-dir ./index` first.")
    st.stop()

# Load mapping and FAISS
with open(MAPPING_PATH, "rb") as f:
    mapping = pickle.load(f)
index = faiss.read_index(FAISS_PATH)

# Load models
@st.cache_resource(show_spinner=False)
def load_models():
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    # QA pipeline (downloads model the first time)
    qa = pipeline("question-answering", model=QA_MODEL_NAME, tokenizer=QA_MODEL_NAME, device=-1)
    return embedder, qa

embedder, qa_pipeline = load_models()

def retrieve(query, k=5):
    q_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_vec, k)
    docs = []
    for idx in I[0]:
        if idx == -1:
            continue
        entry = mapping.get(int(idx))
        if entry:
            combined = entry["question"] + " " + entry["answer"]
            docs.append({"id": int(idx), "text": combined, "question": entry["question"], "answer": entry["answer"]})
    return docs

def answer_with_qa(query, docs):
    # Run QA for each retrieved doc, collect scores, pick best
    candidates = []
    for doc in docs:
        try:
            res = qa_pipeline(question=query, context=doc["text"])
            # result: {'score': float, 'start': int, 'end': int, 'answer': str}
            candidates.append({"id": doc["id"], "answer": res.get("answer", ""), "score": float(res.get("score", 0)), "source_q": doc["question"]})
        except Exception as e:
            # skip if a doc is too long or model fails
            continue
    if not candidates:
        return None, []
    # pick best by score
    best = max(candidates, key=lambda x: x["score"])
    # also return top 3 retrieved as sources
    top_sources = sorted(candidates, key=lambda x: x["score"], reverse=True)[:3]
    return best, top_sources

# UI controls
query = st.text_input("Ask a medical question (e.g., 'What are early symptoms of diabetes?')", "")
k = st.slider("How many documents to retrieve (k)", min_value=1, max_value=8, value=5)

if st.button("Ask") and query.strip():
    with st.spinner("Retrieving relevant FAQ entries..."):
        docs = retrieve(query, k=k)
    if not docs:
        st.warning("No relevant FAQ entries found.")
    else:
        st.subheader("Top retrieved snippets (grounding):")
        for d in docs[:k]:
            st.write(f"- (ID {d['id']}) Q: {d['question']}")
            st.write(f"  A: {d['answer']}")
        with st.spinner("Running local QA model to extract best answer..."):
            best, top_sources = answer_with_qa(query, docs)
        if best is None or best["score"] < 0.08:
            # fallback: return the top FAQ answer (conservative)
            fallback = docs[0]["answer"]
            st.subheader("Answer (fallback from FAQ):")
            st.write(fallback)
            st.caption("Confidence low for extractive QA — showing top FAQ answer as fallback.")
        else:
            st.subheader("Answer (extracted from retrieved context):")
            st.write(best["answer"])
            st.markdown(f"**Confidence:** {best['score']:.2f}")
            st.markdown("**Sources:**")
            for s in top_sources:
                st.write(f"- ID {s['id']} — Q: {s['source_q']} (score={s['score']:.2f})")
