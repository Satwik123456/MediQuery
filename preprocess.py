import argparse
import pandas as pd
import os
from typing import List, Dict

def inspect_dataset(path: str):
    df = pd.read_csv(path)
    print(f"Rows: {len(df)}, Columns: {list(df.columns)}")
    print(df.head(5).to_markdown())

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50):
    # simple whitespace-based chunking in characters
    if not isinstance(text, str):
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = max(0, end - overlap)
    return chunks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    args = parser.parse_args()
    inspect_dataset(args.input)
