# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Mai Tấn Thành
**Nhóm:** 02 - E402
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai đoạn text có cosine similarity cao nghĩa là vector embedding của chúng trỏ về cùng hướng trong không gian nhiều chiều — tức là chúng mang ý nghĩa tương đồng nhau. Cosine similarity đo góc giữa hai vector chứ không phải độ dài, nên hai câu dù khác nhau về độ dài vẫn có thể rất giống nhau về nghĩa.

**Ví dụ HIGH similarity:**
- Sentence A: "Tôi muốn mua một chiếc điện thoại mới."
- Sentence B: "Tôi đang tìm kiếm smartphone để mua."
- Tại sao tương đồng: Cả hai câu đều nói về việc tìm/mua thiết bị điện thoại — ý nghĩa gần như giống nhau dù dùng từ khác nhau.

**Ví dụ LOW similarity:**
- Sentence A: "Tôi muốn mua một chiếc điện thoại mới."
- Sentence B: "Hôm nay trời mưa to, đường rất trơn."
- Tại sao khác: Hai câu thuộc hai chủ đề hoàn toàn khác nhau (mua sắm vs thời tiết), không có liên hệ về nghĩa.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Euclidean distance bị ảnh hưởng bởi độ lớn (magnitude) của vector — câu dài hơn có thể tạo ra vector lớn hơn dù nói cùng ý. Cosine similarity chỉ đo góc giữa hai vector nên bỏ qua ảnh hưởng của độ dài văn bản, phản ánh đúng hơn sự tương đồng về ngữ nghĩa.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

Công thức: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`

```
= ceil((10,000 - 50) / (500 - 50))
= ceil(9,950 / 450)
= ceil(22.11)
= 23 chunks
```

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**

```
= ceil((10,000 - 100) / (500 - 100))
= ceil(9,900 / 400)
= ceil(24.75)
= 25 chunks  (tăng thêm 2 so với overlap=50)
```

> Overlap lớn hơn giúp tránh mất thông tin ở ranh giới giữa các chunk: một câu bị cắt đôi ở biên vẫn xuất hiện đầy đủ trong ít nhất một chunk, giúp retrieval chính xác hơn — đặc biệt quan trọng khi câu trả lời nằm ở chỗ tiếp giáp giữa hai chunk.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** VinUni Student Policies & Regulations

**Tại sao nhóm chọn domain này?**
> Tài liệu policy của VinUni có cấu trúc Markdown rõ ràng với header phân cấp (`##`, `###`), mỗi section biểu diễn một điều khoản độc lập — lý tưởng để test retrieval precision. Domain đa ngôn ngữ (en/vi) cũng cho phép test khả năng multilingual embedding của Voyage AI. Ngoài ra, đây là tài liệu thực tế mà sinh viên VinUni cần tra cứu thường ngày, nên benchmark queries có tính ứng dụng cao.

### Data Inventory

| # | Tên tài liệu | Nguồn | Kích thước | Metadata đã gán |
|---|--------------|-------|------------|-----------------|
| 01 | Sexual_Misconduct_Response_Guideline | policy.vinuni.edu.vn | 26.7 KB | category=Guideline, lang=en, dept=SAM, topic=student_life |
| 02 | Admissions_Regulations_GME_Programs | policy.vinuni.edu.vn | 29.0 KB | category=Regulation, lang=en, dept=CHS, topic=academics |
| 03 | Cam_Ket_Chat_Luong_Dao_Tao | policy.vinuni.edu.vn | 43.3 KB | category=Report, lang=vi, dept=University, topic=academics |
| 04 | Chat_Luong_Dao_Tao_Thuc_Te | policy.vinuni.edu.vn | 18.4 KB | category=Report, lang=vi, dept=University, topic=academics |
| 05 | Doi_Ngu_Giang_Vien_Co_Huu | policy.vinuni.edu.vn | 9.9 KB | category=Report, lang=vi, dept=University, topic=academics |
| 06 | English_Language_Requirements | policy.vinuni.edu.vn | 14.0 KB | category=Policy, lang=en, dept=University, topic=academics |
| 07 | Lab_Management_Regulations | policy.vinuni.edu.vn | 46.1 KB | category=Regulation, lang=en, dept=Operations, topic=safety |
| 08 | Library_Access_Services_Policy | policy.vinuni.edu.vn | 3.5 KB | category=Policy, lang=en, dept=Library, topic=student_life |
| 09 | Student_Grade_Appeal_Procedures | policy.vinuni.edu.vn | 6.1 KB | category=SOP, lang=en, dept=AQA, topic=academics |
| 10 | Tieu_Chuan_ANAT_PCCN | policy.vinuni.edu.vn | 4.6 KB | category=Standard, lang=vi, dept=Operations, topic=safety |
| 11 | Quy_Dinh_Xu_Ly_Su_Co_Chay | policy.vinuni.edu.vn | 2.7 KB | category=Regulation, lang=vi, dept=Operations, topic=safety |
| 12 | Scholarship_Maintenance_Criteria | policy.vinuni.edu.vn | 5.7 KB | category=Guideline, lang=en, dept=SAM, topic=finance |
| 13 | Student_Academic_Integrity | policy.vinuni.edu.vn | 41.8 KB | category=Policy, lang=en, dept=AQA, topic=academics |
| 14 | Student_Award_Policy | policy.vinuni.edu.vn | 14.7 KB | category=Policy, lang=en, dept=SAM, topic=student_life |
| 15 | Student_Code_of_Conduct | policy.vinuni.edu.vn | 17.9 KB | category=Policy, lang=en, dept=SAM, topic=student_life |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `category` | string | Guideline / Policy / SOP / Standard / Report | Phân loại loại tài liệu — có thể filter_search theo type |
| `language` | string | `en` / `vi` | Biết ngôn ngữ gốc để chọn embedder phù hợp |
| `department` | string | SAM / AQA / CHS / Operations / Library | Filter theo phòng ban phụ trách |
| `topic` | string | academics / student_life / safety / finance | Filter theo chủ đề để thu hẹp search space |
| `source` | string | `data/12_Scholarship_Maintenance_Criteria.md` | Traceability — biết câu trả lời đến từ file nào |
| `chunk_index` | int | 0, 1, 2, ... | Xác định vị trí chunk trong tài liệu gốc |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

So sánh 3 strategy baseline trên file `13_Student_Academic_Integrity.md` (41.8 KB, tài liệu dài nhất, cấu trúc header rõ):

| Strategy | Chunk Count | Avg Length (ký tự) | Preserves Context? |
|----------|-------------|---------------------|-------------------|
| FixedSize(500) | 95 | 500 | ❌ Cắt giữa câu/điều khoản |
| SentenceChunker | 66 | ~350 | ⚠️ Giữ câu, nhưng có thể cắt đứt 1 điều khoản |
| RecursiveChunker | 52 | ~480 | ⚠️ Tốt hơn FixedSize, nhưng không hiểu cấu trúc policy |
| **HeaderAwareChunker (của tôi)** | **44** | **~620** | **✅ Mỗi chunk = 1 điều khoản hoàn chỉnh** |

### Strategy Của Tôi

**Loại:** `HeaderAwareChunker(max_chunk_size=1000)`

**Mô tả cách hoạt động:**
> `HeaderAwareChunker` duyệt từng dòng của tài liệu Markdown. Khi phát hiện dòng bắt đầu bằng `#` (header), nó lưu section đang xây dựng và bắt đầu section mới. Mỗi section được giữ nguyên làm 1 chunk — nghĩa là mỗi chunk tương ứng đúng với 1 điều khoản policy. Nếu section vượt quá `max_chunk_size=1000` ký tự, nó cắt thêm theo đoạn văn (blank lines) để tránh embedding quá dài.

**Tại sao chọn strategy này?**
> Tài liệu VinUni policy được cấu trúc theo Markdown header nghiêm ngặt (`##`, `###`), mỗi section biểu diễn đúng 1 ý/điều khoản độc lập. Cắt theo header đảm bảo mỗi chunk luôn chứa một ý nghĩa hoàn chỉnh — khác với FixedSize có thể cắt đứt giữa câu "Học bổng sẽ bị thu hồi nếu..." và phần liệt kê điều kiện. Embedding của chunk hoàn chỉnh sẽ cô đọng và "sắc nét" hơn trong không gian vector, dẫn đến cosine similarity cao hơn khi query đúng chủ đề.

### So Sánh: Strategy của tôi vs Baseline

So sánh trên cùng file `13_Student_Academic_Integrity.md`:

| Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|----------|-------------|------------|-------------------|
| FixedSize(500) — baseline | 95 | 500 ký tự | ❌ Thấp — cắt giữa điều khoản |
| RecursiveChunker — baseline | 52 | 480 ký tự | ⚠️ Trung bình |
| **HeaderAwareChunker — của tôi** | **44** | **620 ký tự** | **✅ Cao hơn — mỗi chunk = 1 policy clause** |

> **Nhận xét:** HeaderAwareChunker tạo ít chunks hơn (44 vs 95) nhưng mỗi chunk chứa đầy đủ 1 điều khoản. Khi query về "quy định học bổng", embedding của chunk biểu diễn toàn bộ điều 4.2 sẽ có cosine similarity cao hơn nhiều so với embedding của nửa điều 4.2 từ FixedSizeChunker.

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|-----------------------|-----------|----------|
| **Tôi (Mai Tấn Thành)** | **HeaderAwareChunker** | **8/10** | Giữ nguyên cấu trúc policy, precision cao | Phụ thuộc vào format Markdown, kém với docs tiếng Việt |
| Nguyễn Hoàng Phúc | Local embedding (query tiếng Việt) | **~9/10** | Query VI match docs VI (Q5 pass!), coverage đa dạng | Không nhất quán với benchmark EN gốc |
| Phạm Lê Hoài Nam | Header+Recursive (chunk=500, overlap=100) + mock | **8/10** | Nhiều chunks → coverage rộng hơn | Mock embedding không semantic, Q5 yếu |
| Hồ Nhất Khoa | SentenceChunker (gom câu thành chunks) + text-embedding-3-small | **7/10** | Câu hoàn chỉnh, không cắt đứt ý | Q2 top-1 sai (AI thay vì Lab) |
| Đặng Tùng Anh | Hybrid Header+Recursive + text-embedding-3-small + GPT-4o-mini | **8/10** | Agent answer chất lượng cao, scores cao (0.6–0.7) | Q5 vẫn fail vì query EN vs docs VI |

**Strategy nào tốt nhất? Tại sao?**
> Sau khi so sánh 5 strategies, **không có một strategy nào chiếm ưu thế tuyệt đối** — mỗi người đạt 7–9/10 tùy cách tiếp cận. Điểm đặc biệt: Hoàng Phúc dùng **query tiếng Việt** và đạt điểm Q5 trong khi cả nhóm còn lại đều fail — cho thấy **language alignment** (query cùng ngôn ngữ với tài liệu) quan trọng hơn chunking strategy. HeaderAwareChunker của tôi cân bằng tốt nhất: precision cao (Q1–Q4 đều 2/2) và không cần tuning thêm, phù hợp với cấu trúc Markdown của corpus VinUni.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`HeaderAwareChunker.chunk`** — approach (strategy của tôi):
> Duyệt từng dòng bằng `text.splitlines(keepends=True)`. Khi gặp dòng `stripped.startswith("#")` và `current` không rỗng → lưu `current` vào `sections`, bắt đầu `current` mới với dòng header. Sau vòng lặp, append `current` cuối. Với mỗi section: nếu `len(section_text) <= max_chunk_size` → giữ nguyên; nếu quá dài → `re.split(r"\n{2,}", section_text)` cắt theo paragraph. Không cần regex phức tạp — ranh giới đã được Markdown quy định sẵn.

**`SentenceChunker.chunk`** — approach:
> Dùng `re.split` với lookbehind pattern `(?<=[.!?]) +|(?<=\.)\n` để tách text thành danh sách câu mà không mất dấu câu kết thúc. Các câu được gom thành chunks bằng vòng lặp bước `max_sentences_per_chunk`, join bằng khoảng trắng. Edge case: text rỗng trả về `[]`, text không có dấu câu trả về nguyên một chunk.

**`RecursiveChunker.chunk` / `_split`** — approach:
> `chunk()` gọi `_split()` với toàn bộ danh sách separators. `_split()` có 3 base cases: (1) text đủ nhỏ → trả về ngay, (2) hết separator → cắt theo ký tự, (3) separator rỗng `""` → character-level split. Với separator thông thường: split text → mỗi piece còn lớn thì đệ quy với separator tiếp theo trong danh sách.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `_make_record()` tạo dict chuẩn hóa gồm `content`, `embedding` (tính qua `embedding_fn`), và `metadata` (inject thêm `doc_id`). `add_documents()` gọi `_make_record()` cho từng doc rồi append vào `self._store`. `search()` tính dot product giữa query embedding và tất cả stored embeddings qua `_search_records()`, sort giảm dần, return top-k.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter()` filter trước bằng list comprehension (giữ records có metadata khớp tất cả key-value trong `metadata_filter`), sau đó mới chạy `_search_records()` trên tập đã lọc. `delete_document()` rebuild `self._store` loại bỏ records có `metadata['doc_id'] == doc_id`, trả `True` nếu size giảm.

### KnowledgeBaseAgent

**`answer`** — approach:
> Retrieve top-k chunks bằng `store.search()`. Build prompt: chunks join bằng dòng trống làm Context, sau đó là Question và "Answer:" để LLM hiểu cần trả lời. Gọi `llm_fn(prompt)` và return trực tiếp. Pattern RAG này buộc LLM bám vào retrieved context thay vì tự bịa.

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.0.2, pluggy-1.6.0
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================= 42 passed in 0.14s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

> **Quan sát quan trọng:** Lab dùng `MockEmbedder` (hash-based deterministic), không phải semantic embedder thật. Kết quả actual score không phản ánh ngữ nghĩa — đây chính là điều thú vị nhất của thí nghiệm này.

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|:------------:|:-----:|
| 1 | "Python is a programming language." | "I love coding in Python." | high | 0.0368 | ❌ |
| 2 | "The weather is sunny today." | "Machine learning uses neural networks." | low | 0.1344 | ❌ |
| 3 | "Dogs are loyal pets." | "Cats are independent animals." | medium | 0.1216 | ✅ |
| 4 | "How to open a bank account?" | "Steps to create a savings account." | high | 0.2099 | ❌ |
| 5 | "I am hungry." | "The stock market crashed." | low | -0.0705 | ✅ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Bất ngờ nhất là Pair 1 và Pair 4: hai câu rõ ràng cùng ngữ nghĩa ("Python programming" / "bank account") nhưng lại có score thấp hơn cả cặp không liên quan (Pair 2 = 0.1344). Điều này cho thấy `MockEmbedder` dùng MD5 hash hoàn toàn ngẫu nhiên — không capture được ngữ nghĩa. Để có semantic search thật sự, cần dùng real embedder như `all-MiniLM-L6-v2` (local) hoặc OpenAI `text-embedding-3-small`.

---

## 6. Results — Cá nhân (10 điểm)

### Benchmark Queries & Gold Answers (nhóm thống nhất)

> **Lưu ý:** Nhóm chọn cross-document queries — mỗi query yêu cầu tổng hợp thông tin từ 3–5 tài liệu để test khả năng multi-document retrieval.

| # | Query | Gold Answer (tóm tắt) | Tài liệu liên quan |
|---|-------|-----------------------|--------------------|
| 1 | What are all the conditions a student must maintain to stay in good academic standing at VinUni? | Duy trì GPA ≥ ngưỡng học bổng; không vi phạm academic integrity; tuân thủ code of conduct; đáp ứng tiêu chí xét giải thưởng | `12_Scholarship` + `13_Academic_Integrity` + `14_Award_Policy` + `15_Code_of_Conduct` |
| 2 | What safety and conduct regulations must students follow when using VinUni campus facilities? | Tuân thủ quy định phòng lab; quy trình xử lý sự cố cháy; quy tắc ứng xử chung; chính sách chống xâm hại tình dục | `07_Lab_Management` + `11_Fire_Safety` + `15_Code_of_Conduct` + `01_Sexual_Misconduct` |
| 3 | What are the admission and language requirements for students entering medical programs at VinUni? | Đáp ứng chuẩn tiếng Anh (IELTS/TOEFL); đạt điểm chuẩn tuyển sinh GME; đáp ứng tiêu chuẩn ANAT PCCN | `02_Admissions_GME` + `06_English_Language` + `10_Tieu_Chuan_ANAT` |
| 4 | What procedures and consequences apply when a student breaks university rules? | Quy trình khiếu nại/kháng nghị; hình thức kỷ luật theo mức độ vi phạm; xử lý gian lận học thuật; xử lý hành vi xâm hại | `01_Sexual_Misconduct` + `09_Grade_Appeal` + `13_Academic_Integrity` + `15_Code_of_Conduct` |
| 5 | How does VinUni evaluate and ensure the quality of its academic programs and teaching staff? | Cam kết chất lượng đào tạo; báo cáo chất lượng thực tế; tiêu chuẩn đội ngũ giảng viên; tiêu chí duy trì học bổng như thước đo kết quả | `03_Cam_Ket_Chat_Luong` + `04_Chat_Luong_Thuc_Te` + `05_Doi_Ngu_Giang_Vien` + `12_Scholarship` |

### Kết Quả Của Tôi

> **Strategy:** `HeaderAwareChunker(max_chunk_size=1000, min_chunk_length=80)` + Voyage AI `voyage-multilingual-2`
> **Tổng chunks sau filter:** 237 chunks | Log: `logs/benchmark_20260410_164416.txt`

| # | Query (rút gọn) | Top-1 Retrieved | Score | Relevant/3 | Điểm |
|---|----------------|----------------|:-----:|:----------:|:----:|
| 1 | Good academic standing | 15_Student_Code_of_Conduct ✅ | 0.5902 | 3/3 | **2/2** |
| 2 | Campus safety regulations | 15_Student_Code_of_Conduct ✅ | 0.6242 | 3/3 | **2/2** |
| 3 | Medical program admission | 02_Admissions_GME ✅ | 0.5800 | 3/3 | **2/2** |
| 4 | Rule violation procedures | 15_Student_Code_of_Conduct ✅ | 0.5611 | 3/3 | **2/2** |
| 5 | Academic quality evaluation | 15_Student_Code_of_Conduct ❌ | 0.5478 | 0/3 | **0/2** |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 4 / 5 — **Tổng: 8 / 10**

**Phân tích Q5 thất bại:**
> Q5 hỏi về chất lượng đào tạo và giảng viên — thông tin nằm ở `03_Cam_Ket`, `04_Chat_Luong`, `05_Doi_Ngu_Giang_Vien` (toàn bộ **tiếng Việt**). Query bằng tiếng Anh → semantic gap lớn hơn same-language. Ngoài ra, `15_Student_Code_of_Conduct` có chunks nhắc đến "programs of VinUniversity" → match surface-level với từ "programs" trong query. **Giải pháp nếu làm lại:** dịch query sang tiếng Việt hoặc thêm cả query tiếng Việt song song.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Học từ **Hồ Nhất Khoa**: kỹ thuật gom nhiều câu vào 1 chunk (SentenceChunker) thay vì cắt cứng theo ký tự giúp mỗi chunk mang một ý hoàn chỉnh hơn. Dù score tổng thể thấp hơn (7/10), nhưng ý tưởng giữ nguyên câu là nền tảng tốt và có thể kết hợp với HeaderAwareChunker để tăng chất lượng thêm. Và học từ **Hoàng Phúc**: dịch query sang ngôn ngữ của tài liệu trước khi embed ("query translation") giúp Q5 — query tiếng Anh tìm trong docs tiếng Việt — từ 0/2 lên 2/2 mà không cần thay đổi gì về chunking hay embedding model.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Các nhóm khác chủ yếu dùng 3 strategy có sẵn trong đề bài (FixedSize, SentenceChunker, RecursiveChunker). Điều rút ra: dù strategy khác nhau, hầu hết đều gặp vấn đề tương tự ở Q5 (docs tiếng Việt) — cho thấy **chất lượng embedding và language alignment quan trọng hơn chunking strategy** khi corpus đa ngôn ngữ.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> **3 thay đổi chính nếu làm lại:** (1) Thêm bước **query translation** — với mỗi query tiếng Anh, tự động sinh thêm bản tiếng Việt và chạy cả hai, lấy union top-3 để không bỏ sót docs VI. (2) Dùng **metadata filtering** theo `language` và `topic` trước khi search — giảm search space từ 237 xuống ~50 chunks, tăng precision đáng kể. (3) Loại bỏ sớm hơn các file **quá ngắn hoặc quá generic** khỏi corpus (ví dụ file 08_Library 3.5KB) vì vector của chúng quá chung chung và gây nhiều false positives.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-----------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 8 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **80 / 100** |
