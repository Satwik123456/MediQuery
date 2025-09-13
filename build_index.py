import os
import pickle
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

def build_index(input_csv, index_dir):
    # Load CSV
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip().str.lower()

    # Extract texts
    texts = df["question"].astype(str) + " " + df["answer"].astype(str)

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts.tolist(), show_progress_bar=True)

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Ensure index directory exists
    os.makedirs(index_dir, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))

    # Save mapping (id → question+answer)
    mapping = {i: {"question": df["question"][i], "answer": df["answer"][i]} for i in range(len(df))}
    with open(os.path.join(index_dir, "mapping.pkl"), "wb") as f:
        pickle.dump(mapping, f)

    print(f"✅ Index built successfully! Saved to {index_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--index-dir", type=str, required=True, help="Directory to save index")
    args = parser.parse_args()

    build_index(args.input, args.index_dir)
