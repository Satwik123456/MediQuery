# app.py
import os
import pickle
import faiss
import numpy as np
import subprocess
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --- Auto-build FAISS index if missing ---
INDEX_DIR = "./index"
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
MAPPING_PATH = os.path.join(INDEX_DIR, "mapping.pkl")

if not os.path.exists(FAISS_PATH):
    st.info("⏳ Hang on, your data is loading… This may take a minute.")
    subprocess.run(["python", "build_index.py", "--input", "train.csv", "--index-dir", INDEX_DIR])
    st.success("✅ FAISS index built successfully!")

# Config
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
QA_MODEL_NAME = "distilbert-base-cased-distilled-squad"

# --- Red & Blue Theme UI ---
st.set_page_config(page_title="MediQuery 🎉", layout="centered", page_icon="💊")

st.markdown("""
<div style='background: linear-gradient(to right, #FF0000, #1E90FF); padding:20px; border-radius:20px; box-shadow:3px 3px 10px rgba(0,0,0,0.2)'>
    <h1 style='color:white; text-align:center;'>MediQuery — Local RAG Medical FAQ Chatbot 💉</h1>
    <p style='color:white; text-align:center; font-size:16px;'>
    Ask your medical questions and get answers from the FAQ knowledge base!
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Check index files
if not os.path.exists(FAISS_PATH) or not os.path.exists(MAPPING_PATH):
    st.error("Index not found. Run `python build_index.py --input train.csv --index-dir ./index` first.")
    st.stop()

# Load mapping and FAISS
with open(MAPPING_PATH, "rb") as f:
    mapping = pickle.load(f)
index = faiss.read_index(FAISS_PATH)

# Load models
@st.cache_resource(show_spinner=False)
def load_models():
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    qa = pipeline("question-answering", model=QA_MODEL_NAME, tokenizer=QA_MODEL_NAME, device=-1)
    return embedder, qa

embedder, qa_pipeline = load_models()

# --- Functions ---
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
    candidates = []
    for doc in docs:
        try:
            res = qa_pipeline(question=query, context=doc["text"])
            candidates.append({"id": doc["id"], "answer": res.get("answer", ""), "score": float(res.get("score", 0)), "source_q": doc["question"]})
        except Exception:
            continue
    if not candidates:
        return None, []
    best = max(candidates, key=lambda x: x["score"])
    top_sources = sorted(candidates, key=lambda x: x["score"], reverse=True)[:3]
    return best, top_sources

# --- UI Controls ---
query = st.text_input("📝 Ask a medical question:", "")
k = st.slider("🔎 How many FAQs to retrieve?", min_value=1, max_value=8, value=5)

if st.button("🚀 Ask!") and query.strip():
    with st.spinner("✨ Retrieving relevant FAQ entries..."):
        docs = retrieve(query, k=k)
    
    if not docs:
        st.warning("⚠️ No relevant FAQ entries found!")
    else:
        st.subheader("💡 Top retrieved snippets:")
        for d in docs[:k]:
            with st.expander(f"Q: {d['question']}"):
                st.markdown(f"""
                <div style='background-color:#FF0000; color:white; padding:10px; border-radius:10px; margin-bottom:5px; box-shadow:2px 2px 5px rgba(0,0,0,0.1)'>
                    <b>A:</b> {d['answer']}
                </div>
                """, unsafe_allow_html=True)
        
        with st.spinner("🧠 Running QA model..."):
            best, top_sources = answer_with_qa(query, docs)
        
        if best is None or best["score"] < 0.08:
            fallback = docs[0]["answer"]
            st.markdown(f"""
            <div style='background-color:#FF0000; color:white; padding:15px; border-radius:15px; box-shadow:3px 3px 10px rgba(0,0,0,0.2)'>
                <h3>🩺 Answer (FAQ fallback):</h3>
                <p>{fallback}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            confidence_percent = int(best["score"]*100)
            st.markdown(f"""
            <div style='background-color:#1E90FF; color:white; padding:15px; border-radius:15px; box-shadow:3px 3px 10px rgba(0,0,0,0.2)'>
                <h3>💊 Answer (from retrieved context):</h3>
                <p>{best['answer']}</p>
                <p><b>Confidence:</b> {best['score']:.2f}</p>
                <div style='background-color:#eee; width:100%; border-radius:5px;'>
                    <div style='background-color:#00008B; width:{confidence_percent}%; height:15px; border-radius:5px'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📚 Sources:")
            for s in top_sources:
                st.markdown(f"- Q: {s['source_q']} (score={s['score']:.2f})")
