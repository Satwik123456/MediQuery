import os
import json
import numpy as np

def load_index(index_dir):
    import faiss
    index = faiss.read_index(os.path.join(index_dir, 'faiss_index.idx'))
    with open(os.path.join(index_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    return index, metadata

def embed_texts(texts):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    resp = client.embeddings.create(model='text-embedding-3-small', input=texts)
    return [np.array(r.embedding, dtype='float32') for r in resp.data]

def retrieve(query, index, metadata, top_k=5):
    v = embed_texts([query])[0]
    v = v.reshape(1, -1)
    D, I = index.search(v, top_k)
    results = []
    for idx in I[0]:
        results.append(metadata[idx])
    return results

def generate_answer(query, retrieved, system_prompt=None):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    # craft a prompt that includes retrieved context
    context = '\n\n'.join([f"Q: {r['question']}\nA: {r['answer']}" for r in retrieved])
    messages = [
        {'role': 'system', 'content': system_prompt or 'You are a helpful medical FAQ assistant. Answer only using the provided context.'},
        {'role': 'user', 'content': f"Using only the context below, answer the question:\n\nContext:\n{context}\n\nQuestion: {query}"}
    ]
    resp = client.chat.completions.create(model='gpt-4o-mini', messages=messages, max_tokens=300)
    return resp.choices[0].message['content']
