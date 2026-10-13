import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import shutil
import asyncio

load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder, SentenceTransformer, util

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
DATA_DIR   = os.getenv("DATA_DIR",   "./data")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

app = FastAPI(title="AIDD ChatBot API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model loading (done once at startup) ──────────────────────────────────────
print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

print("Loading reranker model...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Loading eval embedding model...")
eval_embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading vector store...")
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model
)

print("Loading Gemini LLM...")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    streaming=True,
)

PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
You are an AI legal due diligence assistant.

Your task is to analyze the provided legal context and identify risks related to the user's question.

Follow these rules strictly:
- Base your answer only on the provided context.
- Do not assume or invent information.
- Keep the answer clear and well-structured.
- Provide output in the exact format below.

Output format:

Risk Level: <Low / Medium / High>

Evidence:
- <Short bullet point from document>
- <Short bullet point from document>
- <Short bullet point from document>

Analysis:
<2-3 sentence explanation of the risk and its implications.>

If no relevant information is found, say:
Risk Level: Unknown
Evidence:
- No relevant clause found.

Context:
{context}

Question:
{question}
"""
)

parser = StrOutputParser()

# ── Helpers ───────────────────────────────────────────────────────────────────
def rerank_documents(query: str, docs: list, top_n: int = 3) -> list:
    if not docs:
        return []
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_n]]


def context_relevance_score(query: str, docs: list) -> float:
    if not docs:
        return 0.0
    q_emb = eval_embed_model.encode(query, convert_to_tensor=True)
    scores = []
    for doc in docs:
        d_emb = eval_embed_model.encode(doc.page_content, convert_to_tensor=True)
        scores.append(util.cos_sim(q_emb, d_emb).item())
    return round(sum(scores) / len(scores), 3)


def get_retriever(k: int = 10):
    return vectorstore.as_retriever(search_kwargs={"k": k})


def list_ingested_docs() -> list:
    try:
        col = vectorstore._collection
        results = col.get(include=["metadatas"])
        sources = set()
        for meta in results["metadatas"]:
            src = meta.get("source", "")
            if src:
                sources.add(os.path.basename(src))
        return sorted(sources)
    except Exception:
        return []

# ── Schemas ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3


class QueryResponse(BaseModel):
    answer: str
    relevance_score: float
    sources: List[str]

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def get_documents():
    """List all ingested documents."""
    return {"documents": list_ingested_docs()}


@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    """Upload and ingest a PDF into the vector store."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = os.path.join(DATA_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    loader = PyPDFLoader(save_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vectorstore.add_documents(chunks)

    return {
        "message": f"Successfully ingested '{file.filename}'",
        "chunks": len(chunks),
        "filename": file.filename,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Query the legal documents and get a structured risk analysis."""
    retriever = get_retriever(k=10)
    retrieved = retriever._get_relevant_documents(req.question, run_manager=None)

    if not retrieved:
        return QueryResponse(
            answer="Risk Level: Unknown\n\nEvidence:\n- No documents found in the knowledge base.",
            relevance_score=0.0,
            sources=[],
        )

    top_docs = rerank_documents(req.question, retrieved, top_n=req.top_k)
    score = context_relevance_score(req.question, top_docs)
    context = "\n\n".join(doc.page_content for doc in top_docs)
    sources = list({os.path.basename(doc.metadata.get("source", "")) for doc in top_docs if doc.metadata.get("source")})

    chain = PROMPT_TEMPLATE | llm | parser
    answer = chain.invoke({"context": context, "question": req.question})

    return QueryResponse(answer=answer, relevance_score=score, sources=sources)


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """Stream the LLM response token by token."""
    retriever = get_retriever(k=10)
    retrieved = retriever._get_relevant_documents(req.question, run_manager=None)
    top_docs = rerank_documents(req.question, retrieved, top_n=req.top_k)
    score = context_relevance_score(req.question, top_docs)
    context = "\n\n".join(doc.page_content for doc in top_docs)
    sources = list({os.path.basename(doc.metadata.get("source", "")) for doc in top_docs if doc.metadata.get("source")})

    chain = PROMPT_TEMPLATE | llm | parser

    async def event_stream():
        # First emit metadata
        import json
        meta = json.dumps({"relevance_score": score, "sources": sources})
        yield f"data: __META__{meta}\n\n"
        async for chunk in chain.astream({"context": context, "question": req.question}):
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0)
        yield "data: __DONE__\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
