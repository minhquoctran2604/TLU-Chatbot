# Hệ thống hỏi đáp thông minh dựa trên LightRAG cho dữ liệu Trường Đại học Thủy Lợi

> Đồ án tốt nghiệp — Trần Văn Minh Quốc (64HTTT1), GVHD: TS. Lê Thị Tú Kiên
> Khoa Công nghệ thông tin, Trường Đại học Thủy Lợi

Hệ thống chatbot hỏi đáp trên kho tài liệu học tập (bài giảng, giáo trình) của Trường Đại học Thủy Lợi, xây dựng trên khung **Sinh tăng cường truy hồi dựa trên đồ thị tri thức** (Graph-based RAG) [LightRAG](https://github.com/HKUDS/LightRAG). Đồ án bổ sung một **chế độ truy hồi theo đồ thị (graph ego-walk) tự thiết kế** và một **khung đánh giá benchmark** so sánh nhiều chế độ truy hồi.

> Bản README gốc của framework LightRAG (HKU) được giữ tại [`README_LightRAG_upstream.md`](./README_LightRAG_upstream.md).

---

## 1. Đóng góp của đồ án

Phần đóng góp riêng (so với LightRAG gốc) nằm ở các tệp:

| Tệp | Đóng góp |
|---|---|
| `lightrag/operate.py` | **Chế độ graph (ego-walk + PPR)** — truy hồi theo cấu trúc đồ thị: gieo hạt từ từ khóa, lan truyền dòng chảy (flow) bằng BFS, tinh chỉnh bằng Personalized PageRank, cộng dồn flow đa nguồn để thưởng nút cầu nối |
| `lightrag/prompt.py` | Prompt trích xuất tùy chỉnh cho slide tiếng Việt: lọc nhãn bố cục slide, bộ loại thực thể cố định, lượt trích xuất bổ sung (gleaning) |
| `eval/` | Khung đánh giá tự động: sinh câu hỏi đa loại, chạy benchmark đa chế độ, chấm cặp bằng LLM-as-judge, dashboard kết quả |

### Chế độ graph (ego-walk) — 5 bước

1. **Gieo hạt:** nhúng từ khóa mức thấp của câu hỏi, chọn K thực thể gần nhất trên kho vector làm seed.
2. **BFS lan flow:** dòng chảy giảm dần qua mỗi bước theo hệ số α và chia cho bậc nút để hạn chế nút bậc cao; flow từ nhiều seed được cộng dồn (theo định lý tuyến tính của PPR).
3. **Tinh chỉnh PPR:** chạy lặp Personalized PageRank trên đồ thị con tới hội tụ, thưởng nút cầu cách seed nhiều bước.
4. **Lọc và xếp hạng:** lọc cạnh theo từ khóa, xếp hạng thực thể theo `cosine + λ·flow`.
5. **Thu chunk:** lấy các đoạn văn bản nguồn của thực thể và cạnh đã chọn làm ngữ cảnh.

---

## 2. Năm chế độ truy hồi được so sánh

| Chế độ | Mô tả |
|---|---|
| `bm25` | Baseline từ vựng (Okapi BM25), lập chỉ mục độc lập trong `eval/` |
| `naive` | Tìm kiếm vector thuần |
| `hybrid` | Truy hồi hai mức trên đồ thị (từ khóa thực thể + từ khóa quan hệ) |
| `mix` | Hợp nhất truy hồi đồ thị và vector |
| `graph` | **Ego-walk + PPR (đóng góp của đồ án)** |

---

## 3. Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e ".[api]"

cp env.example .env               # điền cấu hình LLM / embedding / cơ sở dữ liệu
```

Thành phần sử dụng: PostgreSQL + pgvector (vector), NetworkX/GraphML (đồ thị),
mô hình nhúng `BAAI/bge-m3`, mô hình sinh Gemma 3n qua Ollama, Cohere rerank (tùy chọn).

---

## 4. Chạy hệ thống

```bash
lightrag-server                                     # production
uvicorn lightrag.api.lightrag_server:app --reload   # development
```

Truy vấn theo từng chế độ qua tham số `mode` (`bm25` / `naive` / `hybrid` / `mix` / `graph`).

---

## 5. Đánh giá (benchmark)

Toàn bộ quy trình nằm trong thư mục `eval/`:

```bash
cd eval
python gen_queries.py         # sinh câu hỏi đơn-môn theo từng loại
python gen_2hop.py            # sinh câu hỏi suy luận bắc cầu liên môn (2-hop)
python run_benchmark.py       # gửi mỗi câu hỏi tới 5 chế độ, thu câu trả lời
python evaluate_pairwise.py   # chấm cặp bằng LLM-as-judge, tổng hợp điểm Borda
python build_dashboard.py     # dựng dashboard kết quả
```

Bộ câu hỏi và phương pháp chấm:
- 576 câu hỏi thuộc 5 loại (factoid, relational, broad, aggregate, 2-hop liên môn).
- Chấm cặp ẩn danh bằng mô hình ngôn ngữ làm giám khảo, xáo trộn vị trí để giảm thiên lệch.
- Hai chỉ số: số lần xếp hạng nhất (win count) và thứ hạng trung bình (mean rank).

---

## 6. Kết quả chính

Trên benchmark 576 câu × 5 chế độ = 2.880 câu trả lời:

- Các chế độ dựa trên đồ thị (graph, hybrid, mix) vượt trội hai baseline từ vựng (bm25) và vector thuần (naive).
- **Chế độ graph đạt số lần xếp hạng nhất cao nhất (167/576)** và dẫn đầu ở 4/5 loại câu hỏi theo win count, đặc biệt mạnh ở nhóm câu hỏi suy luận bắc cầu liên môn (2-hop).

---

## 7. Cấu trúc thư mục chính

```
lightrag/        # mã nguồn framework (đóng góp ở operate.py, prompt.py)
  ├─ operate.py  # logic truy hồi 5 chế độ + graph ego-walk + PPR
  ├─ prompt.py   # prompt trích xuất tùy chỉnh tiếng Việt
  ├─ kg/         # các backend lưu trữ (NetworkX, PostgreSQL, ...)
  └─ llm/        # các bộ kết nối mô hình ngôn ngữ
eval/            # khung đánh giá: sinh câu hỏi, benchmark, chấm cặp, dashboard
api/             # máy chủ FastAPI + WebUI
```

---

## 8. Ghi nhận

Đồ án xây dựng trên framework mã nguồn mở [LightRAG](https://github.com/HKUDS/LightRAG)
(Guo và cộng sự, HKU, 2024). Phần đóng góp riêng của đồ án được nêu ở Mục 1.
Giấy phép gốc của LightRAG được giữ trong tệp [`LICENSE`](./LICENSE).
