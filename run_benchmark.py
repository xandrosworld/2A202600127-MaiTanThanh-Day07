"""
run_benchmark.py — Chạy 5 benchmark queries, load từ cache nếu có.
Kết quả in ra terminal và lưu vào logs/benchmark_results.txt
"""
import os, pickle, sys, datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv(override=False)

from src.chunking import HeaderAwareChunker
from src.embeddings import VOYAGE_EMBEDDING_MODEL, VoyageEmbedder, _mock_embed
from src.models import Document
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent

# ─── Config ────────────────────────────────────────────────────────────────────
DATA_FILES = [
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

BENCHMARK_QUERIES = [
    {
        "id": 1,
        "short": "Good academic standing",
        "full": "What are all the conditions a student must maintain to stay in good academic standing at VinUni?",
        "expected_docs": ["12_Scholarship_Maintenance_Criteria", "13_Student_Academic_Integrity",
                          "14_Student_Award_Policy", "15_Student_Code_of_Conduct"],
    },
    {
        "id": 2,
        "short": "Campus safety regulations",
        "full": "What safety and conduct regulations must students follow when using VinUni campus facilities?",
        "expected_docs": ["07_Lab_Management_Regulations", "11_Quy_Dinh_Xu_Ly_Su_Co_Chay",
                          "15_Student_Code_of_Conduct", "01_Sexual_Misconduct_Response_Guideline"],
    },
    {
        "id": 3,
        "short": "Medical program admission",
        "full": "What are the admission and language requirements for students entering medical programs at VinUni?",
        "expected_docs": ["02_Admissions_Regulations_GME_Programs", "06_English_Language_Requirements",
                          "10_Tieu_Chuan_ANAT_PCCN"],
    },
    {
        "id": 4,
        "short": "Rule violation procedures",
        "full": "What procedures and consequences apply when a student breaks university rules?",
        "expected_docs": ["01_Sexual_Misconduct_Response_Guideline", "09_Student_Grade_Appeal_Procedures",
                          "13_Student_Academic_Integrity", "15_Student_Code_of_Conduct"],
    },
    {
        "id": 5,
        "short": "Academic quality evaluation",
        "full": "How does VinUni evaluate and ensure the quality of its academic programs and teaching staff?",
        "expected_docs": ["03_Cam_Ket_Chat_Luong_Dao_Tao", "04_Chat_Luong_Dao_Tao_Thuc_Te",
                          "05_Doi_Ngu_Giang_Vien_Co_Huu", "12_Scholarship_Maintenance_Criteria"],
    },
]

CACHE_PATH = Path("data/.embedding_cache.pkl")

# ─── Load store ────────────────────────────────────────────────────────────────
def get_embedder():
    """Get the correct embedder from .env."""
    provider = os.getenv("EMBEDDING_PROVIDER", "mock").strip().lower()
    if provider == "voyage":
        try:
            embedder = VoyageEmbedder(model_name=os.getenv("VOYAGE_EMBEDDING_MODEL", VOYAGE_EMBEDDING_MODEL))
            print(f"Embedder: {embedder._backend_name}")
            return embedder
        except Exception as e:
            print(f"Voyage failed: {e} — using mock")
    print("Embedder: mock")
    return _mock_embed


def load_store():
    embedder = get_embedder()  # Always load the right embedder first

    # Try cache first (stored vectors must match embedder!)
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            records = cached.get("records", [])
            if records:
                print(f"✅ Loaded {len(records)} chunks from cache (instant!)\n")
                store = EmbeddingStore(collection_name="benchmark", embedding_fn=embedder)
                store._store = records
                return store
        except Exception as e:
            print(f"Cache load failed: {e} — re-indexing...")

    # Re-index
    print("No valid cache — embedding with Voyage AI (may take ~60s)...")
    chunker = HeaderAwareChunker(max_chunk_size=1000, min_chunk_length=80)
    store = EmbeddingStore(collection_name="benchmark", embedding_fn=embedder)
    chunk_docs = []
    for fp in DATA_FILES:
        p = Path(fp)
        if not p.exists():
            continue
        chunks = chunker.chunk(p.read_text(encoding="utf-8"))
        for idx, txt in enumerate(chunks):
            chunk_docs.append(Document(
                id=f"{p.stem}_chunk_{idx:03d}",
                content=txt,
                metadata={"source": p.name, "doc_id": p.stem, "chunk_index": idx},
            ))
    print(f"Indexed {len(chunk_docs)} chunks")
    store.add_documents(chunk_docs)

    # Save cache
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"records": store._store}, f)
    print("Cache saved.\n")
    return store

# ─── Run benchmark ─────────────────────────────────────────────────────────────
def run_benchmark(store):
    lines = []
    def p(s=""):
        print(s)
        lines.append(s)

    p("=" * 70)
    p(f"VinUni RAG Benchmark — HeaderAwareChunker + Voyage AI")
    p(f"Ngày chạy: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p(f"Tổng chunks: {len(store._store)}")
    p("=" * 70)

    total_relevant = 0

    for q in BENCHMARK_QUERIES:
        p(f"\n{'─'*70}")
        p(f"Q{q['id']}: {q['short']}")
        p(f"Query: {q['full']}")
        p(f"Expected docs: {', '.join(d.split('_', 1)[1] for d in q['expected_docs'])}")

        results = store.search(q["full"], top_k=3)
        relevant_in_top3 = 0

        p("\nTop-3 Retrieved Chunks:")
        for i, r in enumerate(results, 1):
            doc_id = r["metadata"].get("doc_id", "")
            is_relevant = any(exp in doc_id for exp in q["expected_docs"])
            if is_relevant:
                relevant_in_top3 += 1
            marker = "✅" if is_relevant else "❌"
            p(f"  #{i} {marker} score={r['score']:.4f}  source={r['metadata'].get('source','?')}")
            p(f"      preview: {r['content'][:120].replace(chr(10),' ')}...")

        score = 2 if (relevant_in_top3 >= 1 and any(exp in results[0]["metadata"].get("doc_id","") for exp in q["expected_docs"])) else \
                1 if relevant_in_top3 >= 1 else 0
        total_relevant += score
        p(f"\nRelevant in top-3: {relevant_in_top3}/3 → Score: {score}/2")

    p(f"\n{'='*70}")
    p(f"TỔNG ĐIỂM RETRIEVAL: {total_relevant} / 10")
    p("=" * 70)

    return lines

# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    store = load_store()
    lines = run_benchmark(store)

    # Save log
    Path("logs").mkdir(exist_ok=True)
    log_path = f"logs/benchmark_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n📄 Log saved → {log_path}")
