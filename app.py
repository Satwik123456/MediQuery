# app.py
import csv
import pickle
import hashlib
import unicodedata
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from streamlit.components.v1 import html

# Optional GPU check (safe if torch not installed)
try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "index"
FAISS_PATH = INDEX_DIR / "faiss.index"
MAPPING_PATH = INDEX_DIR / "mapping.pkl"
CACHE_PATH = INDEX_DIR / "embed_cache.pkl"   # cache: text_hash -> vector
META_PATH = INDEX_DIR / "meta.pkl"           # stores file sig, keys, model, etc.
DEFAULT_CSV = BASE_DIR / "train.csv"         # fixed CSV (no UI)

# ---------- Config ----------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
QA_MODEL_NAME = "distilbert-base-cased-distilled-squad"
ANSWER_CHARS = 200  # how many answer characters to embed (0 = question only)
DEFAULT_BATCH = 128 if (TORCH_OK and torch.cuda.is_available()) else 64

# ---------- Page ----------
st.set_page_config(page_title="MediQuery 🎉", layout="centered", page_icon="💊")

# ---------- Styles ----------
st.markdown("""
<style>
/* Animated gradient on the app container (near black • blue • red) */
.stApp {
    background: linear-gradient(-45deg, #070708, #0b1233, #5b0012, #070708);
    background-size: 400% 400%;
    animation: gradientBG 18s ease infinite;
}
[data-testid="stAppViewContainer"], [data-testid="stHeader"], .block-container {
    background: transparent !important;
}

/* Sidebar: glassy panel matching theme */
[data-testid="stSidebar"] {
    background: rgba(6,10,24,0.55) !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .stSlider, [data-testid="stSidebar"] .stCheckbox { color: #f3f6ff !important; }

/* Readable text on dark */
.stApp, .stApp p, .stApp li, .stApp label, .stApp span,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 { color: #f3f6ff !important; }

/* Inputs on dark bg */
div[data-baseweb="input"] input, textarea {
    color: #eef2ff !important;
    background: rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
}

/* Title (old MediQuery red->blue) */
.title-card {
    background: linear-gradient(to right, #FF0000, #1E90FF);
    padding: 20px;
    border-radius: 20px;
    box-shadow: 3px 3px 12px rgba(0,0,0,0.45);
}

/* Cards with slight glass effect */
.answer-card, .conf-card, .fallback-card {
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
}

/* Answer card */
.answer-card {
    background: rgba(30,144,255,0.12);
    border: 1px solid rgba(30,144,255,0.35);
    color: #ffffff;
    padding: 12px 14px;
    border-radius: 14px;
    max-width: 640px;
    margin: 8px 0 6px 0;
}

/* Confidence card (base dark blue; ramp set inline) */
.conf-card {
    background: rgba(10, 35, 107, 0.60);
    border: 1px solid rgba(16, 55, 160, 0.85);
    color: #e6f0ff;
    padding: 10px 12px;
    border-radius: 12px;
    max-width: 420px;
    margin-top: 8px;
    font-size: 0.92rem;
}
.conf-bar {
    background: rgba(255,255,255,0.22);
    width: 100%;
    height: 8px;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 6px;
}
.conf-bar > span { display: block; height: 100%; }

/* Fallback card (red tone) */
.fallback-card {
    background: rgba(220,20,60,0.18);
    border: 1px solid rgba(220,20,60,0.35);
    color: #ffffff;
    padding: 12px 14px;
    border-radius: 14px;
    max-width: 640px;
    margin: 8px 0 6px 0;
}

/* Expander header readable */
[data-testid="stExpander"] div[role="button"] { color: #fff !important; }

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown("""
<div class='title-card'>
    <h1 style='color:white; text-align:center; margin:0;'>MediQuery — Local RAG Medical FAQ Chatbot 💉</h1>
    <p style='color:white; text-align:center; font-size:16px; margin:6px 0 0 0;'>
    Ask your medical questions and get answers from the FAQ knowledge base!
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ---------- Helpers ----------
def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return " ".join(s.strip().split())

def conf_color(score: float) -> str:
    s = max(0.0, min(1.0, float(score)))
    if s < 0.3:
        return "#b00020"  # red
    elif s < 0.6:
        return "#ffb300"  # amber
    else:
        return "#1E90FF"  # blue

def darker(hex_color: str, factor=0.75) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = int(r * factor); g = int(g * factor); b = int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"

def play_bubble(volume=0.5):
    v = max(0.0, min(0.9, float(volume)))
    js = """
        <script>(function(){
          try{
            const AC = window.AudioContext || window.webkitAudioContext;
            const ctx = new AC();
            if (ctx.state === 'suspended') { ctx.resume(); }
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            const bp = ctx.createBiquadFilter();
            bp.type = 'bandpass';
            bp.frequency.setValueAtTime(900, ctx.currentTime);
            bp.Q.setValueAtTime(8, ctx.currentTime);
            osc.type = 'sine';
            osc.frequency.setValueAtTime(220, ctx.currentTime);
            osc.frequency.exponentialRampToValueAtTime(1200, ctx.currentTime + 0.04);
            osc.frequency.exponentialRampToValueAtTime(70, ctx.currentTime + 0.22);
            gain.gain.setValueAtTime(0.0001, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(%s, ctx.currentTime + 0.02);
            gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.24);
            const clickBuf = ctx.createBuffer(1, 256, ctx.sampleRate);
            const data = clickBuf.getChannelData(0); data[0] = 1;
            const clickSrc = ctx.createBufferSource();
            clickSrc.buffer = clickBuf;
            const clickGain = ctx.createGain();
            clickGain.gain.setValueAtTime(%s, ctx.currentTime);
            osc.connect(bp).connect(gain).connect(ctx.destination);
            clickSrc.connect(clickGain).connect(ctx.destination);
            osc.start();
            clickSrc.start(ctx.currentTime + 0.02);
            osc.stop(ctx.currentTime + 0.26);
          }catch(e){ console.log('Audio blocked:', e); }
        })();</script>
    """ % (v, v/2.0)
    html(js, height=0, width=0)

def row_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

# ---------- Lazy model loading ----------
@st.cache_resource(show_spinner=False)
def get_device():
    if TORCH_OK and torch.cuda.is_available():
        return "cuda", 0
    return "cpu", -1

@st.cache_resource(show_spinner=False)
def get_embedder():
    device_str, _ = get_device()
    from sentence_transformers import SentenceTransformer
    with st.spinner("Loading embedding model… (first time may take a minute)"):
        return SentenceTransformer(EMBED_MODEL_NAME, device=device_str)

@st.cache_resource(show_spinner=False)
def get_qa_pipeline():
    _, dev_id = get_device()
    from transformers import pipeline
    with st.spinner("Loading QA model… (first time may take a minute)"):
        return pipeline("question-answering", model=QA_MODEL_NAME, tokenizer=QA_MODEL_NAME, device=dev_id)

# ---------- Cache / Meta ----------
def load_cache() -> dict:
    if CACHE_PATH.exists():
        with CACHE_PATH.open("rb") as f:
            return pickle.load(f)
    return {}

def save_cache(cache: dict):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("wb") as f:
        pickle.dump(cache, f)

def load_meta():
    if META_PATH.exists():
        with META_PATH.open("rb") as f:
            return pickle.load(f)
    return None

def save_meta(keys_list, embed_model, file_sig, answer_chars, row_count):
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with META_PATH.open("wb") as f:
        pickle.dump(
            {
                "keys": keys_list,
                "embed_model": embed_model,
                "file_sig": file_sig,
                "answer_chars": answer_chars,
                "row_count": row_count,
            },
            f,
        )

# ---------- KB preparation (auto, no UI) ----------
def read_qa_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")
        name_map = {n.lower(): n for n in reader.fieldnames}
        if "question" not in name_map or "answer" not in name_map:
            raise ValueError("CSV must contain 'question' and 'answer' columns.")
        qs, ans = [], []
        for row in reader:
            qs.append(normalize_text(row.get(name_map["question"]) or ""))
            ans.append(normalize_text(row.get(name_map["answer"]) or ""))
    return qs, ans

def load_index_files_into_state():
    if MAPPING_PATH.exists() and FAISS_PATH.exists():
        with MAPPING_PATH.open("rb") as f:
            st.session_state.mapping = pickle.load(f)
        st.session_state.faiss_index = faiss.read_index(str(FAISS_PATH))

def ensure_knowledge_base():
    if not DEFAULT_CSV.exists():
        st.error(f"CSV not found: {DEFAULT_CSV}")
        return

    # Quick path: if files on disk are present and up-to-date, just load
    stat = DEFAULT_CSV.stat()
    curr_sig = (stat.st_mtime, stat.st_size)
    meta = load_meta()
    if (
        meta
        and meta.get("file_sig") == curr_sig
        and meta.get("embed_model") == EMBED_MODEL_NAME
        and meta.get("answer_chars") == ANSWER_CHARS
        and MAPPING_PATH.exists()
        and FAISS_PATH.exists()
    ):
        load_index_files_into_state()
        return

    # Build or update index (incremental)
    with st.spinner("Preparing knowledge base…"):
        questions, answers = read_qa_csv(DEFAULT_CSV)
        embed_texts = [
            (f"{q} {a[:ANSWER_CHARS]}".strip() if ANSWER_CHARS > 0 else q)
            for q, a in zip(questions, answers)
        ]
        keys = [row_key(t) for t in embed_texts]
        n = len(embed_texts)
        if n == 0:
            st.error("CSV has no rows.")
            return

        cache = load_cache()
        to_encode_idx = []
        vectors = [None] * n
        for i, k in enumerate(keys):
            if k in cache:
                vectors[i] = cache[k]
            else:
                to_encode_idx.append(i)

        if to_encode_idx:
            embedder = get_embedder()
            for pos in range(0, len(to_encode_idx), DEFAULT_BATCH):
                idx_batch = to_encode_idx[pos : pos + DEFAULT_BATCH]
                text_batch = [embed_texts[i] for i in idx_batch]
                emb = embedder.encode(text_batch, batch_size=DEFAULT_BATCH, convert_to_numpy=True).astype("float32")
                for j, i in enumerate(idx_batch):
                    vectors[i] = emb[j]
                    cache[keys[i]] = emb[j]
            save_cache(cache)
        else:
            # all from cache
            for i in range(n):
                if vectors[i] is None:
                    vectors[i] = cache[keys[i]]

        X = np.vstack(vectors)
        dim = X.shape[1]
        faiss_index = faiss.IndexFlatL2(dim)
        faiss_index.add(X)

        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(faiss_index, str(FAISS_PATH))
        mapping = {i: {"question": questions[i], "answer": answers[i]} for i in range(n)}
        with MAPPING_PATH.open("wb") as f:
            pickle.dump(mapping, f)

        save_meta(keys, EMBED_MODEL_NAME, curr_sig, ANSWER_CHARS, n)

        st.session_state.mapping = mapping
        st.session_state.faiss_index = faiss_index

# Prepare KB once (fast if unchanged)
ensure_knowledge_base()

# ---------- Sidebar (no refresh) ----------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    play_sound = st.checkbox("🔔 Bubble sound", value=True)
    volume = st.slider("Volume", 0.0, 1.0, 0.6, 0.05)

    st.markdown("##### Retrieval")
    k = st.slider("How many FAQs to retrieve?", min_value=1, max_value=8, value=5)

# ---------- Retrieval + QA ----------
def retrieve(query, top_k=5):
    if "faiss_index" not in st.session_state or "mapping" not in st.session_state:
        return []
    embedder = get_embedder()
    q_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    D, I = st.session_state.faiss_index.search(q_vec, top_k)
    docs = []
    for idx in I[0]:
        if idx == -1:
            continue
        entry = st.session_state.mapping.get(int(idx))
        if entry:
            combined = entry["question"] + " " + entry["answer"]
            docs.append({"id": int(idx), "text": combined, "question": entry["question"], "answer": entry["answer"]})
    return docs

def answer_with_qa(query, docs):
    qa = get_qa_pipeline()
    candidates = []
    for doc in docs:
        try:
            res = qa(question=query, context=doc["text"])
            candidates.append({"id": doc["id"], "answer": res.get("answer", ""), "score": float(res.get("score", 0))})
        except Exception:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda x: x["score"])

# ---------- Main controls ----------
st.markdown("---")
query = st.text_input("📝 Ask a medical question:", "")

# Hints if KB missing
if ("faiss_index" not in st.session_state or "mapping" not in st.session_state) and query == "":
    st.info(f"No knowledge base yet. Ensure {DEFAULT_CSV.name} exists next to app.py.")

# ---------- Action ----------
if st.button("🚀 Ask!") and query.strip():
    if "faiss_index" not in st.session_state or "mapping" not in st.session_state:
        st.error("Knowledge base not ready. Add train.csv and restart.")
    else:
        with st.spinner("✨ Retrieving relevant FAQ entries..."):
            docs = retrieve(query, top_k=k)

        if not docs:
            st.warning("⚠️ No relevant FAQ entries found!")
        else:
            st.subheader("💡 Top retrieved snippets:")
            for d in docs[:k]:
                snippet_html = """
                <div style='background-color:#FF0000; color:white; padding:10px; border-radius:10px; margin-bottom:5px; box-shadow:2px 2px 5px rgba(0,0,0,0.1)'>
                    <b>A:</b> %s
                </div>
                """ % (d["answer"])
                st.markdown(snippet_html, unsafe_allow_html=True)

            with st.spinner("🧠 Running QA model..."):
                best = answer_with_qa(query, docs)

            if best is None or best["score"] < 0.08:
                fallback_html = """
                <div class='fallback-card'>
                    <p style='margin:0;'>%s</p>
                </div>
                """ % (docs[0]["answer"])
                st.markdown(fallback_html, unsafe_allow_html=True)
                if play_sound:
                    play_bubble(volume=volume)
            else:
                conf = float(best["score"])
                conf_pct = max(0, min(100, int(conf * 100)))
                c = conf_color(conf)
                c_dark = darker(c, 0.7)

                answer_html = """
                <div class='answer-card'>
                    <p style='margin:0;'>%s</p>
                </div>
                """ % (best["answer"])
                st.markdown(answer_html, unsafe_allow_html=True)

                conf_html = """
                <div class='conf-card' style='border-color:%s;'>
                    <div><b>Confidence:</b> %.2f</div>
                    <div class='conf-bar'>
                        <span style='width:%s%%; background: linear-gradient(90deg, %s 0%%, %s 100%%);'></span>
                    </div>
                </div>
                """ % (c_dark, conf, conf_pct, c_dark, c)
                st.markdown(conf_html, unsafe_allow_html=True)

                if play_sound:
                    play_bubble(volume=volume)