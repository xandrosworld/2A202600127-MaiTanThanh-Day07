from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import HeaderAwareChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

SAMPLE_FILES = [
    "data/01_Sexual_Misconduct_Response_Guideline.md",
    "data/02_Admissions_Regulations_GME_Programs.md",
    "data/03_Cam_Ket_Chat_Luong_Dao_Tao.md",
    "data/04_Chat_Luong_Dao_Tao_Thuc_Te.md",
    "data/05_Doi_Ngu_Giang_Vien_Co_Huu.md",
    "data/06_English_Language_Requirements.md",
    "data/07_Lab_Management_Regulations.md",
    "data/08_Library_Access_Services_Policy.md",
    "data/09_Student_Grade_Appeal_Procedures.md",
    "data/10_Tieu_Chuan_ANAT_PCCN.md",
    "data/11_Quy_Dinh_Xu_Ly_Su_Co_Chay.md",
    "data/12_Scholarship_Maintenance_Criteria.md",
    "data/13_Student_Academic_Integrity.md",
    "data/14_Student_Award_Policy.md",
    "data/15_Student_Code_of_Conduct.md",
]

METADATA_MAP = {
    "01_Sexual_Misconduct_Response_Guideline":  {"category": "Guideline",   "language": "en", "department": "SAM",        "topic": "student_life"},
    "02_Admissions_Regulations_GME_Programs":  {"category": "Regulation",   "language": "en", "department": "CHS",        "topic": "academics"},
    "03_Cam_Ket_Chat_Luong_Dao_Tao":           {"category": "Report",       "language": "vi", "department": "University",  "topic": "academics"},
    "04_Chat_Luong_Dao_Tao_Thuc_Te":           {"category": "Report",       "language": "vi", "department": "University",  "topic": "academics"},
    "05_Doi_Ngu_Giang_Vien_Co_Huu":            {"category": "Report",       "language": "vi", "department": "University",  "topic": "academics"},
    "06_English_Language_Requirements":         {"category": "Policy",       "language": "en", "department": "University",  "topic": "academics"},
    "07_Lab_Management_Regulations":            {"category": "Regulation",   "language": "en", "department": "Operations",  "topic": "safety"},
    "08_Library_Access_Services_Policy":        {"category": "Policy",       "language": "en", "department": "Library",     "topic": "student_life"},
    "09_Student_Grade_Appeal_Procedures":       {"category": "SOP",          "language": "en", "department": "AQA",         "topic": "academics"},
    "10_Tieu_Chuan_ANAT_PCCN":                 {"category": "Standard",     "language": "vi", "department": "Operations",  "topic": "safety"},
    "11_Quy_Dinh_Xu_Ly_Su_Co_Chay":           {"category": "Regulation",   "language": "vi", "department": "Operations",  "topic": "safety"},
    "12_Scholarship_Maintenance_Criteria":      {"category": "Guideline",    "language": "en", "department": "SAM",         "topic": "finance"},
    "13_Student_Academic_Integrity":            {"category": "Policy",       "language": "en", "department": "AQA",         "topic": "academics"},
    "14_Student_Award_Policy":                  {"category": "Policy",       "language": "en", "department": "SAM",         "topic": "student_life"},
    "15_Student_Code_of_Conduct":               {"category": "Policy",       "language": "en", "department": "SAM",         "topic": "student_life"},
}


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        extra_meta = METADATA_MAP.get(path.stem, {})
        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={
                    "source": str(path),
                    "extension": path.suffix.lower(),
                    "domain": "VinUni Student Policies",
                    **extra_meta,
                },
            )
        )

    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "voyage":
        try:
            from src.embeddings import VOYAGE_EMBEDDING_MODEL, VoyageEmbedder
            embedder = VoyageEmbedder(model_name=os.getenv("VOYAGE_EMBEDDING_MODEL", VOYAGE_EMBEDDING_MODEL))
        except Exception as e:
            print(f"Voyage init failed: {e} — falling back to mock")
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    # Chunk each document with HeaderAwareChunker before embedding
    chunker = HeaderAwareChunker(max_chunk_size=1000)
    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    chunk_docs: list[Document] = []
    for doc in docs:
        chunks = chunker.chunk(doc.content)
        for idx, chunk_text in enumerate(chunks):
            chunk_docs.append(Document(
                id=f"{doc.id}_chunk_{idx:03d}",
                content=chunk_text,
                metadata={
                    **doc.metadata,
                    "doc_id": doc.id,
                    "chunk_index": idx,
                    "domain": "VinUni Student Policies",
                },
            ))
    print(f"  Chunked into {len(chunk_docs)} chunks total (HeaderAwareChunker)")
    store.add_documents(chunk_docs)

    print(f"\nStored {store.get_collection_size()} documents in EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=3))
    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())
