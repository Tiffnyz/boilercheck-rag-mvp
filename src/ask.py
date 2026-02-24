import argparse
import os
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


def dedupe_by_source(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for m in matches:
        md = m.get("metadata", {})
        key = (md.get("url", ""), md.get("source_key", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


def format_sources(matches: List[Dict[str, Any]], max_chars: int = 500) -> str:
    blocks = []
    for i, m in enumerate(matches, start=1):
        md = m["metadata"]
        url = md.get("url", "")
        section = md.get("source_key", "")
        text = md.get("text", "")
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        citation = f"[{url}#{section}]"
        blocks.append(
            f"SOURCE {i}\n"
            f"CITATION: {citation}\n"
            f"TITLE: {md.get('doc_title','')}\n"
            f"SECTION: {section}\n"
            f"TEXT: {text}\n"
        )
    return "\n".join(blocks)


def retrieve(index, model, query: str, top_k: int) -> List[Dict[str, Any]]:
    q_emb = model.encode(query).tolist()
    res = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
    matches = res.get("matches", [])
    matches = dedupe_by_source(matches)[:top_k]
    return matches


def generate_answer_openrouter(prompt: str) -> str:
    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY in .env for --mode generate")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "Purdue Policy RAG MVP",
        },
    )

    resp = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str, help="User question")
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--mode", choices=["retrieve", "generate"], default="retrieve")
    args = parser.parse_args()

    pinecone_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "purdue-policy-index-v3")

    if not pinecone_key:
        raise RuntimeError("Missing PINECONE_API_KEY in .env")

    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(index_name)

    embed_model_name = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
    model = SentenceTransformer(embed_model_name, device="cpu")

    matches = retrieve(index, model, args.question, args.top_k)

    if args.mode == "retrieve":
        print("QUESTION:", args.question)
        print("\nTOP SOURCES:\n")
        for i, m in enumerate(matches, start=1):
            md = m["metadata"]
            print(f"[{i}] {md.get('doc_title','')}")
            print(f"Section: {md.get('source_key','')}")
            print(f"URL: {md.get('url','')}")
            print(f"Excerpt: {md.get('text','')}")
            print()
        return

    sources_text = format_sources(matches, max_chars=450)

    prompt = (
        "You are a Purdue policy assistant. Use ONLY the SOURCES.\n"
        "CITATION RULES:\n"
        "- Cite using the CITATION field exactly.\n"
        "- If the answer is not in the sources, say you cannot confirm and ask 1-2 clarifying questions.\n"
        "- Keep it short (max 4 sentences).\n\n"
        f"SOURCES:\n{sources_text}\n"
        f"QUESTION:\n{args.question}\n"
    )

    answer = generate_answer_openrouter(prompt)
    print(answer)


if __name__ == "__main__":
    main()