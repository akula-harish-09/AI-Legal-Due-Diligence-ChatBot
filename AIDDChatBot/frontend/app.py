import streamlit as st
import requests
import json
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Legal Due Diligence ChatBot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Overall background */
.main { background-color: #0f1117; }

/* Risk badges */
.risk-high   { background:#ff4b4b; color:white; padding:4px 14px; border-radius:20px; font-weight:700; font-size:1rem; }
.risk-medium { background:#ffa500; color:white; padding:4px 14px; border-radius:20px; font-weight:700; font-size:1rem; }
.risk-low    { background:#21c55d; color:white; padding:4px 14px; border-radius:20px; font-weight:700; font-size:1rem; }
.risk-unknown{ background:#6b7280; color:white; padding:4px 14px; border-radius:20px; font-weight:700; font-size:1rem; }

/* Score bar label */
.score-label { font-size:0.85rem; color:#9ca3af; margin-bottom:4px; }

/* Chat bubbles */
.user-bubble {
    background: #1e40af;
    color: white;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
    text-align: right;
}
.bot-bubble {
    background: #1f2937;
    color: #e5e7eb;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 0;
    max-width: 80%;
    border-left: 3px solid #3b82f6;
}

/* Source chips */
.source-chip {
    display: inline-block;
    background: #1e3a5f;
    color: #93c5fd;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    margin: 2px 3px;
    border: 1px solid #2563eb;
}

/* Header */
.aidd-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
    padding: 20px 28px;
    border-radius: 12px;
    margin-bottom: 20px;
}
.aidd-header h1 { color: white; margin: 0; font-size: 1.8rem; }
.aidd-header p  { color: #93c5fd; margin: 4px 0 0; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = []

# ── Helper functions ──────────────────────────────────────────────────────────
def fetch_documents():
    try:
        r = requests.get(f"{API_URL}/documents", timeout=5)
        if r.ok:
            return r.json().get("documents", [])
    except Exception:
        pass
    return []


def ingest_pdf(file_bytes, filename):
    try:
        r = requests.post(
            f"{API_URL}/ingest",
            files={"file": (filename, file_bytes, "application/pdf")},
            timeout=120,
        )
        return r.json() if r.ok else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def query_api(question: str, top_k: int) -> dict:
    try:
        r = requests.post(
            f"{API_URL}/query",
            json={"question": question, "top_k": top_k},
            timeout=120,
        )
        return r.json() if r.ok else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def risk_badge(level: str) -> str:
    lvl = level.strip().lower()
    css = {
        "high": "risk-high",
        "medium": "risk-medium",
        "low": "risk-low",
    }.get(lvl, "risk-unknown")
    return f'<span class="{css}">⚠ {level}</span>'


def parse_risk_level(answer: str) -> str:
    for line in answer.splitlines():
        if line.strip().lower().startswith("risk level:"):
            return line.split(":", 1)[-1].strip()
    return "Unknown"


def render_message(msg: dict):
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        answer = msg["content"]
        meta   = msg.get("meta", {})
        score  = meta.get("relevance_score", None)
        sources = meta.get("sources", [])
        risk   = parse_risk_level(answer)

        col1, col2 = st.columns([4, 1])
        with col1:
            with st.container():
                st.markdown(f'<div class="bot-bubble">', unsafe_allow_html=True)
                # Risk badge
                st.markdown(f"**Risk Level:** {risk_badge(risk)}", unsafe_allow_html=True)
                st.divider()
                st.markdown(answer, unsafe_allow_html=False)
                # Sources
                if sources:
                    chips = "".join(f'<span class="source-chip">📄 {s}</span>' for s in sources)
                    st.markdown(f"**Sources:** {chips}", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            if score is not None:
                st.metric("Relevance", f"{score:.2f}", help="Context relevance score (0–1). Higher = better match.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ AIDD ChatBot")
    st.caption("AI Legal Due Diligence Assistant")
    st.divider()

    # Backend health
    try:
        h = requests.get(f"{API_URL}/health", timeout=3)
        if h.ok:
            st.success("✅ Backend connected")
        else:
            st.error("❌ Backend error")
    except Exception:
        st.error("❌ Backend offline — start `uvicorn main:app` first")

    st.divider()

    # Document upload
    st.markdown("### 📤 Upload Documents")
    uploaded = st.file_uploader(
        "Upload legal PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="PDFs will be ingested into ChromaDB for querying.",
    )
    if uploaded:
        for f in uploaded:
            if st.button(f"Ingest: {f.name}", key=f"ingest_{f.name}"):
                with st.spinner(f"Ingesting {f.name}..."):
                    result = ingest_pdf(f.read(), f.name)
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.success(f"✅ {result['message']} ({result['chunks']} chunks)")
                    st.session_state.ingested_docs = fetch_documents()

    st.divider()

    # Ingested docs list
    st.markdown("### 📚 Knowledge Base")
    if st.button("🔄 Refresh", key="refresh_docs"):
        st.session_state.ingested_docs = fetch_documents()

    docs = st.session_state.ingested_docs or fetch_documents()
    if docs:
        for doc in docs:
            st.markdown(f"- 📄 `{doc}`")
    else:
        st.caption("No documents ingested yet.")

    st.divider()

    # Settings
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Top-K documents for context", 1, 5, 3)
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="aidd-header">
  <h1>⚖️ AI Legal Due Diligence ChatBot</h1>
  <p>Ask legal questions about uploaded contracts and documents. Get structured risk analysis instantly.</p>
</div>
""", unsafe_allow_html=True)

# Render chat history
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.info("👋 Upload a legal document in the sidebar, then ask a question below. For example: *'Are there any termination clauses?'* or *'What are the payment obligations?'*")
    else:
        for msg in st.session_state.messages:
            render_message(msg)

st.divider()

# Input
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Ask a legal question...",
            placeholder="e.g., What are the termination conditions? Are there any penalty clauses?",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("Send ➤", use_container_width=True)

if submitted and user_input.strip():
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("🔍 Analyzing legal documents..."):
        result = query_api(user_input, top_k)

    if "error" in result:
        st.error(f"API Error: {result['error']}")
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "meta": {
                "relevance_score": result.get("relevance_score", 0),
                "sources": result.get("sources", []),
            },
        })
    st.rerun()

# ── Sample questions ──────────────────────────────────────────────────────────
st.markdown("#### 💡 Try asking:")
sample_qs = [
    "What are the termination clauses?",
    "Are there any penalty or indemnification clauses?",
    "What are the confidentiality obligations?",
    "What payment terms are defined?",
    "Are there any automatic renewal clauses?",
]
cols = st.columns(len(sample_qs))
for i, q in enumerate(sample_qs):
    if cols[i].button(q, key=f"sample_{i}", use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.spinner("🔍 Analyzing..."):
            result = query_api(q, top_k)
        if "error" not in result:
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "meta": {
                    "relevance_score": result.get("relevance_score", 0),
                    "sources": result.get("sources", []),
                },
            })
        st.rerun()
