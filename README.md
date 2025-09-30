💊 MediQuery(RAG-based Medical FAQ Chatbot)

A Retrieval-Augmented Generation (RAG) chatbot that answers common medical FAQs using a knowledge base of ~100 questions and answers.
Built with FAISS, Sentence Transformers, OpenAI GPT, and a Streamlit interface.

⚠️ Disclaimer: This chatbot is for educational/demo purposes only. It is not a substitute for professional medical advice.

🚀 Features

Retrieval-Augmented Generation (RAG) pipeline
Embedding-based search using FAISS
Context-aware answers powered by OpenAI GPT
Simple and interactive Streamlit UI
Modular, documented codebase for easy extension

Project Structure
├── app.py               # Streamlit app (chatbot interface)
├── build_index.py       # Script to build FAISS index
├── medical_faqs_100.csv # Dataset (100 medical FAQs)
├── index/               # Saved FAISS index + mapping
├── requirements.txt     # Python dependencies
├── .env                 # API key (not pushed to GitHub)
└── README.md

⚡ Setup & Usage
1️⃣ Clone the repo
git clone https://github.com/yourusername/rag-medical-chatbot.git
cd rag-medical-chatbot

2️⃣ Create a virtual environment
python -m venv .venv

⚠️ On Windows PowerShell, you may need to allow script execution first:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
Then activate the environment:

.\.venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Add OpenAI API key
Create a .env file in the root folder:
OPENAI_API_KEY=your-secret-key-here

5️⃣ Build the index
python build_index.py --input medical_faqs_100.csv --index-dir ./index

6️⃣ Run the chatbot
streamlit run app.py
