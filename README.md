# RAG-based Medical FAQ Chatbot (Demo)

This repository contains an end-to-end, local-demo-ready Retrieval-Augmented Generation (RAG) chatbot that answers medical FAQ queries using a provided dataset (`medical_faqs_100.csv`).

**Key features**
- Preprocesses the provided CSV of medical FAQs.
- Builds a local FAISS vector index from embeddings (instructions included).
- Simple Streamlit UI to ask questions and get answers grounded in the dataset.
- All OpenAI API usage is performed by reading your local `OPENAI_API_KEY` environment variable (never hard-coded).

## What I built (scaffold)
- `preprocess.py` - inspects & chunks the dataset (safe to run offline).
- `build_index.py` - builds embeddings & FAISS index (requires OpenAI key).
- `app.py` - Streamlit application to interact with the chatbot.
- `rag_query.py` - RAG pipeline glue: retrieve top-k + call OpenAI completions.
- `utils.py` - helper utilities.
- `requirements.txt` - Python dependencies.
- `.gitignore` - exclude secrets and env files.

## Dataset
We use the provided `medical_faqs_100.csv`. Please place it in the project root (it already exists for you).

## Setup (local)
1. Create a Python environment (recommended `venv` or `conda`):
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .\.venv\Scripts\activate  # Windows PowerShell
   pip install -r requirements.txt
   ```

2. Set your OpenAI API key (do NOT commit this to GitHub):
   - macOS / Linux:
     ```bash
     export OPENAI_API_KEY="sk-..."
     ```
   - Windows (PowerShell):
     ```powershell
     setx OPENAI_API_KEY "sk-..."
     ```

3. Build the FAISS index (this uses OpenAI embeddings):
   ```bash
   python build_index.py --input medical_faqs_100.csv --index-dir ./index
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Notes about security
- **Do not** add your API key to the repository. Use environment variables as shown above.
- If you accidentally leak a key, revoke it immediately from the OpenAI dashboard.

## Evaluation checklist
- Correctness of RAG pipeline: `build_index.py` + `rag_query.py`.
- Searchable knowledge base: FAISS index of embeddings.
- Quality of responses: LLM is called with retrieved context only.
- End-to-end demo: `app.py` (Streamlit).

## Design choices (short)
- FAISS for local vector store to keep the demo free and portable.
- OpenAI embeddings + GPT for generation (configurable model names).
- Streamlit for quick, user-friendly demo UI.

