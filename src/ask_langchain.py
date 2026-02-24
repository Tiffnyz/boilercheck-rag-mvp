import os
import sys
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

def main(question: str):
    # Embeddings 
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Pinecone
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not index_name:
        raise RuntimeError("Missing PINECONE_INDEX_NAME in .env")

    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        text_key="text",
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    docs = retriever.invoke(question)

    # Format context with URL citations
    context_parts = []
    for i, d in enumerate(docs, start=1):
        md = d.metadata or {}
        section = md.get("source_key", "")
        url = md.get("url", "")
        citation = f"[{url}#{section}]"
        context_parts.append(
            f"SOURCE {i}\nCITATION: {citation}\nTEXT: {d.page_content}\n"
        )
    context = "\n".join(context_parts)

    llm = ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model="openrouter/free",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Purdue policy assistant. Use ONLY the sources provided. "
                "Cite using the CITATION field exactly. If not enough info, say you cannot confirm and ask 1-2 clarifying questions. "
                "Keep it short (max 4 sentences).",
            ),
            ("user", "SOURCES:\n{context}\n\nQUESTION:\n{question}"),
        ]
    )

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    print("\nANSWER:")
    print(response.content)

    print("\nSOURCES USED:\n")
    seen = set()
    for d in docs:
        md = d.metadata or {}
        url = md.get("url", "")
        section_title = md.get("section_title", "")
        subsection_title = md.get("subsection_title", "")
        section_path = section_title if not subsection_title else f"{section_title} > {subsection_title}"
        key = (url, section_path)
        if url and key not in seen:
            print(f"- {url} | {section_path}")
            seen.add(key)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/ask_langchain.py "your question here"')
        raise SystemExit(1)

    main(sys.argv[1])