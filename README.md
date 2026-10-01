# Vietnamese Legal RAG

Trợ lý hỏi đáp pháp luật Việt Nam: **hybrid retrieval (BM25 + dense + reranker)** trên hơn 61K điều luật và **LLM nhỏ fine-tune bằng QLoRA** để trả lời bám ngữ cảnh, **có trích dẫn điều luật** và biết từ chối khi không đủ căn cứ.

Mục tiêu chính là đo được từng thành phần bằng số liệu thật, không dừng ở mức demo.

```
Câu hỏi ─► BM25 (sparse) ─┐
          └► bge-m3 (dense) ┴► RRF fusion ─► bge-reranker (top-30 → top-3)
                                                   │
                         ngữ cảnh đánh số [1][2][3] ▼
                   Qwen2.5-3B-Instruct + LoRA adapter (vLLM)
                                                   │
                     câu trả lời + trích dẫn "Điều 7, 47/2011/TT-BCA"
```

## Dữ liệu

[Zalo AI 2021 – Legal Text Retrieval](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn) (MIT):

- 61,425 điều luật
- 2,432 câu hỏi train và 788 câu hỏi test, mỗi câu có điều luật đúng do người gán nhãn

Bộ dữ liệu **không có câu trả lời mẫu**, nên phần sinh câu trả lời được đánh giá bằng các metric không cần đáp án mẫu (xem bên dưới).

## Kết quả

### Retrieval (tập test, 788 câu hỏi, toàn bộ corpus)

| Phương pháp | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | ms/query |
|---|---|---|---|---|---|---|
| BM25 (âm tiết) | 0.558 | 0.805 | 0.859 | 0.667 | 0.714 | 4 |
| BM25 + bigram âm tiết | 0.626 | 0.852 | 0.890 | 0.726 | 0.766 | 2 |
| Dense (bge-m3) | 0.626 | 0.873 | 0.925 | 0.733 | 0.779 | 17 |
| Hybrid (BM25 + dense, RRF) | 0.687 | 0.895 | 0.938 | 0.779 | 0.818 | 19 |
| **Hybrid + reranker (top-30)** | **0.697** | **0.932** | **0.963** | **0.800** | **0.840** | 410 |
| Dense + reranker (top-30) | 0.698 | 0.919 | 0.949 | 0.795 | 0.832 | 411 |

BM25 chạy trên CPU; dense và reranker chạy trên Kaggle T4, thời gian tính trung bình khi chạy theo batch.

**Số ứng viên đưa vào reranker** (hybrid + reranker):

| Số ứng viên | Recall@5 | Recall@10 | ms/query |
|---|---|---|---|
| 10 | 0.918 | 0.938 | 161 |
| 20 | 0.928 | 0.959 | 287 |
| 30 | **0.932** | **0.963** | 410 |
| 50 | 0.932 | 0.958 | 644 |

**Nhận xét:**
- **Bigram âm tiết** giúp BM25 xấp xỉ việc tách từ tiếng Việt ("xử_phạt", "lao_động") và tăng Recall@1 thêm **6.8 điểm** mà không cần thư viện tách từ. Sau khi có bigram, BM25 ngang dense ở Recall@1 nhưng nhanh hơn khoảng 10 lần.
- **Hybrid hơn hẳn từng phương pháp riêng lẻ** (Recall@1 tăng 6 điểm). BM25 bắt tốt thuật ngữ và số hiệu chính xác, còn dense bắt tốt câu hỏi diễn đạt khác văn bản luật, nên hai cách bổ sung cho nhau.
- **Reranker chủ yếu kéo điều luật đúng vào top-5** (Recall@5 từ 0.895 lên 0.932), còn Recall@1 gần như không đổi. Với RAG như vậy là đủ, vì LLM nhận top-3 làm ngữ cảnh. Đổi lại, độ trễ tăng khoảng 20 lần.
- **Từ 30 ứng viên trở lên thì không tăng thêm**, chỉ tốn thêm thời gian. Mức 20 giữ được gần như toàn bộ chất lượng mà nhanh hơn 30%, nên tôi chọn 20 làm mặc định khi demo.
- **Recall@1 dừng ở khoảng 0.70**, trong khi gần như mọi câu test chỉ có 1 điều luật đúng (mức tối đa đạt được là 0.997). Hướng cải thiện tiếp theo là fine-tune reranker hoặc embedding trên khoảng 2.4K cặp câu hỏi và điều luật đúng của tập train.

### Dữ liệu SFT và fine-tune QLoRA

**Dữ liệu:** 890 mẫu (739 có câu trả lời, 151 mẫu từ chối), distill từ Gemini Flash Lite trên câu hỏi train.
Câu trả lời của teacher chỉ được giữ khi trích dẫn đúng điều luật gold; khoảng 15% bị loại bởi bộ lọc này.
Chia train/val: 846 / 44 mẫu.

**Train:** Qwen2.5-3B-Instruct, QLoRA 4-bit NF4, r=16, 7 lớp chiếu (29.9M tham số, 0.96%), 2 × T4 với DDP, 106 bước (2 epoch), khoảng 4 giờ.

| Bước | Epoch | eval_loss | Độ chính xác token (eval) |
|---|---|---|---|
| 25 | 0.47 | 0.1848 | 95.2% |
| **50** | **0.94** | **0.1651** | **95.5%** |
| 75 | 1.42 | 0.1666 | 95.6% |
| 100 | 1.89 | 0.1680 | 95.6% |

`eval_loss` thấp nhất ở cuối epoch 1 rồi đi ngang, nên dùng checkpoint-50; epoch 2 không cải thiện thêm (1 epoch là đủ cho bộ dữ liệu này).

### Sinh câu trả lời (chế độ oracle, 300 câu test, cùng ngữ cảnh cho mọi mô hình)

| Mô hình | answer_rate | gold_cite_rate | cite_precision | invalid_cite | refusal_acc | faithfulness |
|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct (base) | _TODO_ | | | | | |
| + QLoRA (của mình) | _TODO_ | | | | | |
| Teacher (tham chiếu) | _TODO_ | | | | | |

- **gold_cite_rate:** tỉ lệ câu trả lời có trích dẫn đúng điều luật gold.
- **invalid_cite:** tỉ lệ câu trả lời trích dẫn nguồn không tồn tại trong ngữ cảnh (bịa nguồn).
- **refusal_acc:** tỉ lệ từ chối đúng khi ngữ cảnh không chứa điều luật gold.
- **faithfulness:** chấm bằng LLM-as-judge.

## Cấu trúc

```
src/legalrag/
  data.py        đọc dataset, chunk điều luật dài theo khoản (có overlap), định dạng trích dẫn
  text.py        chuẩn hoá Unicode NFC + tách âm tiết/bigram
  retrieval.py   BM25 (tự cài bằng sparse matrix), Dense (FAISS), Hybrid (RRF), Reranker
  metrics.py     Recall@k, MRR, nDCG
  prompts.py     prompt có ngữ cảnh đánh số, parse trích dẫn, câu từ chối cố định
  sampling.py    ngữ cảnh "gold + hard negatives" cho dữ liệu SFT và đánh giá
  pipeline.py    retrieve → rerank → generate
  factory.py     nạp index, dựng pipeline từ cấu hình
scripts/         download_data, build_index, eval_retrieval, build_sft_data, eval_generation
training/        train_qlora.py (TRL SFTTrainer + PEFT, 4-bit NF4)
app/             FastAPI (/ask, /search, /health) + Gradio UI
tests/           pytest, chạy trên CPU, không cần tải model
```

## Chạy trên Kaggle (khuyến nghị cho các bước cần GPU)

| Notebook | Nội dung | Thời gian |
|---|---|---|
| [notebooks/01_retrieval_kaggle.ipynb](notebooks/01_retrieval_kaggle.ipynb) | Dense index bge-m3, bảng so sánh BM25 / dense / hybrid / reranker | khoảng 1 giờ |
| [notebooks/02_sft_qlora_eval_kaggle.ipynb](notebooks/02_sft_qlora_eval_kaggle.ipynb) | Tạo dữ liệu SFT, train QLoRA, đánh giá base / LoRA / teacher, phân tích lỗi | 2–4 giờ |

Các bước: Kaggle → **Create → Import Notebook** → chọn file `.ipynb` → Settings: **GPU T4 x2**, **Internet On** → thêm Secret `TEACHER_API_KEY` (chỉ notebook 02 cần) → **Run All** → **Save Version** để lưu output. Notebook tự clone repo từ GitHub nên luôn chạy code mới nhất.

## Chạy dự án

### 0. Cài đặt

```bash
python -m venv .venv && .venv\Scripts\activate    # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

### 1. Dữ liệu và index

```bash
python scripts/download_data.py

# Chỉ BM25 (CPU, khoảng 1 phút):
python scripts/build_index.py --bigrams --skip-dense --index-dir indexes/full

# Đầy đủ BM25 + dense: nên chạy trên GPU (Kaggle T4 / Colab), khoảng 106K chunks
python scripts/build_index.py --bigrams --index-dir indexes/full

# Muốn thử nhanh trên CPU: dùng corpus con (gồm mọi điều luật gold + ngẫu nhiên),
# lưu ý số liệu sẽ cao hơn thực tế
python scripts/build_index.py --bigrams --corpus-limit 5000 --index-dir indexes/dev
```

### 2. Đánh giá retrieval

```bash
python scripts/eval_retrieval.py --index-dir indexes/full \
    --methods bm25 dense hybrid hybrid+rerank --output results/retrieval_test.json
```

### 3. Tạo dữ liệu SFT (distillation từ teacher LLM)

Teacher là bất kỳ endpoint nào tương thích OpenAI API. Mỗi mẫu gồm điều luật gold và các hard negative lấy từ BM25, xếp ngẫu nhiên. Teacher viết câu trả lời có trích dẫn. Script tự **lọc bỏ** các câu trả lời không trích dẫn đúng điều luật gold. Khoảng 15% mẫu chỉ có hard negative, với nhãn là câu từ chối, để mô hình học cách không bịa khi retrieval trả về sai.

```bash
export TEACHER_API_KEY=...
python scripts/build_sft_data.py --index-dir indexes/full \
    --teacher-base-url https://api.openai.com/v1 --teacher-model gpt-4o-mini
# -> data/sft/train.jsonl, data/sft/val.jsonl (có thể chạy tiếp nếu bị ngắt)
```

### 4. Fine-tune QLoRA (Kaggle T4 x1 / Colab)

```bash
pip install -r requirements-train.txt
python training/train_qlora.py --model Qwen/Qwen2.5-3B-Instruct --output-dir outputs/qwen-legal-lora
```

Loss chỉ tính trên phần câu trả lời, không tính trên ngữ cảnh dài. Trên T4 script tự dùng fp16.

### 5. Serve và đánh giá base với LoRA

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct --enable-lora \
    --lora-modules legal-lora=outputs/qwen-legal-lora/adapter --max-model-len 4096

python scripts/eval_generation.py --llm-model Qwen/Qwen2.5-3B-Instruct --name base \
    --judge-base-url https://api.openai.com/v1 --judge-model gpt-4o-mini
python scripts/eval_generation.py --llm-model legal-lora --name lora \
    --judge-base-url https://api.openai.com/v1 --judge-model gpt-4o-mini
```

### 6. Demo

```bash
docker compose up --build        # vLLM + API :8080 + UI :7860 (cần GPU NVIDIA)
# hoặc chạy riêng:
uvicorn app.api:app --port 8080
python app/ui.py
```

## Lộ trình 4 tuần

- [x] **Tuần 0:** khung project, BM25 + bigram, metrics, test, số liệu baseline BM25
- [x] **Tuần 1:** dense index bge-m3 trên Kaggle; bảng BM25 / dense / hybrid / +reranker; ablation số ứng viên reranker
- [x] **Tuần 2:** tạo dữ liệu SFT (890 mẫu), kiểm tra mẫu, train QLoRA trên 2 × T4
- [ ] **Tuần 3:** đánh giá base với LoRA với teacher; phân tích lỗi (câu nào sai, vì sao); ablation `neg_ratio`
- [ ] **Tuần 4:** Docker, demo, README hoàn chỉnh, quay GIF demo, đưa lên CV

## Ghi chú kỹ thuật

- **Chunking:** một số điều luật dài hơn 100K ký tự. Nếu không chunk, embedding model sẽ âm thầm cắt mất phần sau. Điểm của một điều luật là điểm của chunk tốt nhất (max-pooling), và chunk đó được đưa vào reranker/LLM.
- **RRF thay vì cộng điểm:** điểm BM25 không bị chặn còn cosine nằm trong [-1, 1]. RRF chỉ dùng thứ hạng nên không cần hiệu chỉnh thang điểm.
- **Đánh giá oracle và pipeline:** ở chế độ oracle, mọi mô hình nhận cùng ngữ cảnh (seed theo câu hỏi), nên so sánh base với LoRA không bị nhiễu bởi retrieval. Chế độ pipeline đo hệ thống end-to-end.
- **Hạn chế:** faithfulness chấm bằng LLM-as-judge nên có sai số. Nên kiểm tra tay một mẫu nhỏ để xem judge có đồng thuận với người chấm không.
