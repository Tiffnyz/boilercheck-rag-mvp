import json
import os
import time
import hashlib
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer


def flatten_policy_json(raw_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []

    for doc in raw_docs:
        doc_id = doc.get("document_id", "")
        doc_title = doc.get("title", "")
        url = doc.get("url", "")
        domain = doc.get("domain", "")
        effective_date = doc.get("effective_date", "")

        for sec in doc.get("sections", []):
            sec_title = sec.get("section_title", "")

            # Case 1: direct text on section
            sec_text = sec.get("text")
            if isinstance(sec_text, str) and sec_text.strip():
                flat.append(
                    {
                        "doc_id": doc_id,
                        "doc_title": doc_title,
                        "url": url,
                        "domain": domain,
                        "effective_date": effective_date,
                        "section_title": sec_title,
                        "subsection_title": "",
                        "text": sec_text,
                    }
                )

            # Case 2: subsections
            for sub in sec.get("subsections", []):
                sub_title = sub.get("section_title", "")
                sub_text = sub.get("text")
                if isinstance(sub_text, str) and sub_text.strip():
                    flat.append(
                        {
                            "doc_id": doc_id,
                            "doc_title": doc_title,
                            "url": url,
                            "domain": domain,
                            "effective_date": effective_date,
                            "section_title": sec_title,
                            "subsection_title": sub_title,
                            "text": sub_text,
                        }
                    )

    return flat


def chunk_text(text: str, max_words: int = 140, overlap: int = 30) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunks.append(" ".join(words[start:end]))
        start += max_words - overlap
    return chunks


def make_chunk_id(doc_id: str, section_title: str, subsection_title: str, chunk_index: int) -> str:
    # Hash ensures all Pinecone IDs are ASCII (otherwise there may be an error)
    key = f"{doc_id}|{section_title}|{subsection_title}|{chunk_index}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return f"{doc_id}__{h}"


def batched(lst: List[Dict[str, Any]], batch_size: int):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


def main():
    load_dotenv()

    data_path = os.getenv("POLICY_JSON_PATH")
    index_name = os.getenv("PINECONE_INDEX_NAME", "purdue-policy-index-v3")
    pinecone_key = os.getenv("PINECONE_API_KEY")

    if not pinecone_key:
        raise RuntimeError("Missing PINECONE_API_KEY in .env")


    # ------------ Embedding model -------------

    embed_model_name = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
    dim = 384  # all-MiniLM-L6-v2
    region = os.getenv("PINECONE_REGION", "us-east-1")
    batch_size = int(os.getenv("UPSERT_BATCH_SIZE", "100"))

    with open(data_path, "r") as f:
        raw_docs = json.load(f)

    flat_docs = flatten_policy_json(raw_docs)
    print("Flat entries:", len(flat_docs))

    all_chunks: List[Dict[str, Any]] = []
    for d in flat_docs:
        doc_id = str(d.get("doc_id", "") or "")
        doc_title = str(d.get("doc_title", "") or "")
        url = str(d.get("url", "") or "")
        domain = str(d.get("domain", "") or "")
        effective_date = str(d.get("effective_date", "") or "")
        section_title = str(d.get("section_title", "") or "")
        subsection_title = str(d.get("subsection_title", "") or "")
        text = str(d.get("text", "") or "")

        for i, chunk in enumerate(chunk_text(text)):
            source_key = section_title if not subsection_title else f"{section_title} > {subsection_title}"
            all_chunks.append(
                {
                    "id": make_chunk_id(doc_id, section_title, subsection_title, i),
                    "text": chunk,
                    "metadata": {
                        "doc_id": doc_id,
                        "doc_title": doc_title,
                        "url": url,
                        "domain": domain,
                        "effective_date": effective_date,
                        "section_title": section_title,
                        "subsection_title": subsection_title,
                        "chunk_index": int(i),
                        "source_key": source_key,
                    },
                }
            )

    print("Total chunks:", len(all_chunks))

    # ---------- Pinecone index setup -----------
    pc = Pinecone(api_key=pinecone_key)
    existing = [i["name"] for i in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=region),
        )
        print("Created index:", index_name)

    index = pc.Index(index_name)
    print("Index ready:", index_name)

    # ---------- Embed + upsert ----------
    model = SentenceTransformer(embed_model_name, device="cpu")
    print("Embedding model loaded:", embed_model_name)

    total = 0
    t0 = time.time()

    for batch in batched(all_chunks, batch_size):
        vectors = []
        for c in batch:
            emb = model.encode(c["text"]).tolist()
            vectors.append(
                {
                    "id": c["id"],
                    "values": emb,
                    "metadata": {**c["metadata"], "text": c["text"]},
                }
            )
        index.upsert(vectors=vectors)
        total += len(vectors)
        print(f"Upserted so far: {total} (elapsed {time.time() - t0:.1f}s)")

    print("Done. Total upserted:", total)
    print("Index stats:", index.describe_index_stats())


if __name__ == "__main__":
    main()