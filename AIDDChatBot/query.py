import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from sentence_transformers import CrossEncoder, SentenceTransformer, util

CHROMA_DIR = "./chroma_db"

# Load reranker model
print("Loading reranker model...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Load embedding model for evaluation
eval_embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def rerank_documents(query, docs, top_n=3):
    """Rerank retrieved documents using cross-encoder."""
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in scored_docs[:top_n]]


def context_relevance_score(query, docs):
    """Compute average cosine similarity between query and docs."""
    query_emb = eval_embed_model.encode(query, convert_to_tensor=True)

    scores = []
    for doc in docs:
        doc_emb = eval_embed_model.encode(doc.page_content, convert_to_tensor=True)
        score = util.cos_sim(query_emb, doc_emb)
        scores.append(score.item())

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def main():
    print("Loading embeddings...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    print("Loading vector database...")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model
    )

    # Retrieve more docs for reranking
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    print("Loading Gemini model...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_template(
        """
You are an AI legal due diligence assistant.

Your task is to analyze the provided legal context and identify risks related to the user's question.

Follow these rules strictly:
- Base your answer only on the provided context.
- Do not assume or invent information.
- Keep the answer short and clear.
- Provide output in the exact format below.

Output format:

Risk Level: <Low / Medium / High>

Evidence:
- <Short bullet point from document>
- <Short bullet point from document>
- <Short bullet point from document>

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

    while True:
        query = input("\nAsk a legal question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        # Step 1: Retrieve
        #retrieved_docs = retriever._get_relevant_documents(query)
        #retrieved_docs = retriever.get_relevant_documents(query)
        retrieved_docs = retriever._get_relevant_documents(query, run_manager=None)

        # Step 2: Rerank
        top_docs = rerank_documents(query, retrieved_docs, top_n=3)

        # Step 3: Context relevance
        score = context_relevance_score(query, top_docs)

        # Step 4: Build context
        context = "\n\n".join(doc.page_content for doc in top_docs)

        # Step 5: LLM answer
        chain = prompt | llm | parser
        answer = chain.invoke({"context": context, "question": query})

        print("\nContext Relevance Score:", round(score, 3))
        print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()
