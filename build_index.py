# build_index.py
import os
import argparse
import pandas as pd
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

def build_index(input_csv, index_dir):
    # Load CSV
    df = pd.read_csv(input_csv)
    # Normalize headers
    df.columns = df.columns.str.strip().str.lower()

    # Accept either 'question'/'answer' or 'Question'/'Answer'
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("CSV must contain 'question' and 'answer' columns (case-insensitive).")

    # Combine question + answer for an embedding context
    texts = (df["question"].astype(str) + " " + df["answer"].astype(str)).tolist()

    # Load embedding model (downloads first time)
    print("Loading SentenceTransformer (this downloads the model if not cached)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create embeddings (numpy array)
    print(f"Generating embeddings for {len(texts)} documents...")
    embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Build FAISS index (L2)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index and mapping
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))

    # Save mapping: id -> {question, answer}
    mapping = {i: {"question": df["question"].iloc[i], "answer": df["answer"].iloc[i]} for i in range(len(df))}
    with open(os.path.join(index_dir, "mapping.pkl"), "wb") as f:
        pickle.dump(mapping, f)

    print("Index built and saved to:", index_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV (medical_faqs_100.csv)")
    parser.add_argument("--index-dir", required=True, help="Directory to save FAISS index and mapping")
    args = parser.parse_args()
    build_index(args.input, args.index_dir)
