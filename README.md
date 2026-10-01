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

Câu trả lời mẫu được distill từ Gemini Flash Lite trên câu hỏi train, chỉ giữ câu trả lời trích dẫn đúng điều luật gold
(khoảng 15% bị bộ lọc loại). Mẫu từ chối (ngữ cảnh chỉ gồm hard negatives từ BM25) tạo tự động, không cần teacher.

| Phiên bản | Dữ liệu | Train | eval_loss tốt nhất |
|---|---|---|---|
| LoRA v1 | 890 mẫu, **17%** từ chối | 2 epoch, 106 bước, khoảng 4 giờ | 0.1651 (bước 50, cuối epoch 1) |
| LoRA v2 | 1.140 mẫu, **35%** từ chối | 1 epoch, 68 bước, khoảng 2.4 giờ | 0.1839 (bước 50) |

Cấu hình chung: Qwen2.5-3B-Instruct, QLoRA 4-bit NF4, r=16 trên 7 lớp chiếu (29.9M tham số, 0.96%), 2 × T4 với DDP.
Ở v1, `eval_loss` thấp nhất ở cuối epoch 1 rồi đi ngang, nên v2 chỉ train 1 epoch. eval_loss của hai bản không so sánh trực tiếp
được vì tập validation khác nhau.

### Sinh câu trả lời (chế độ oracle, 150 câu test, cùng ngữ cảnh cho mọi mô hình)

123 câu có điều luật đúng trong ngữ cảnh, 27 câu chỉ có điều luật sai (mô hình nên từ chối).

| Mô hình | Trả lời | **Trích đúng gold** | Độ chính xác trích dẫn | Bịa nguồn | Không trích dẫn | **Từ chối đúng** | Faithfulness* |
|---|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct (base) | 75.6% | 41.5% | 61.9% | 2.8% | 30.2% | 51.9% | 46.2% (106) |
| + LoRA v1 (17% mẫu từ chối) | 92.7% | **82.1%** | 68.5% | 0% | 0% | 14.8% | 59.1% (137) |
| **+ LoRA v2 (35% mẫu từ chối)** | 86.2% | 78.0% | 72.8% | **0%** | **0%** | 37.0% | 62.6% (123) |
| Teacher: Gemini 3.5 Flash Lite | 77.2% | 76.4% | 83.5% | 0% | 0% | 77.8% | 97.0% (101) |

\* Faithfulness chấm bằng LLM-as-judge (Gemini 3.1 Flash Lite) trên các câu mô hình **trả lời**; số trong ngoặc là số câu được chấm.
Mỗi mô hình từ chối số câu khác nhau nên tập được chấm khác nhau, không so sánh trực tiếp được.

**Nhận xét:**
- **Fine-tune dạy được định dạng trích dẫn:** cả hai bản LoRA gần như gấp đôi tỉ lệ trích đúng điều luật gold so với base
  (41.5% lên 78–82%, ngang hoặc hơn teacher) và loại bỏ hoàn toàn câu thiếu trích dẫn (base: 30%) lẫn trích nguồn không tồn tại.
- **v1 gần như mất khả năng từ chối** (51.9% xuống 14.8%): khi ngữ cảnh chỉ có điều luật sai cùng chủ đề, mô hình vẫn trả lời
  dựa trên điều luật đó. Nguyên nhân là mẫu từ chối chỉ chiếm 17% dữ liệu SFT.
- **v2 tăng tỉ lệ mẫu từ chối lên 35%:** từ chối đúng tăng từ 14.8% lên 37.0%, độ chính xác trích dẫn và faithfulness cũng tăng,
  đổi lại tỉ lệ trích đúng gold giảm nhẹ (82.1% xuống 78.0%). Đây là đánh đổi điển hình giữa trả lời nhiều và từ chối an toàn.
- **Từ chối vẫn kém base và teacher.** Chỉ có 27 câu "nên từ chối" nên số liệu này dao động lớn (mỗi câu tương đương 3.7 điểm phần trăm);
  cần tập test lớn hơn để kết luận chắc chắn.

- **Trích đúng gold:** câu có điều luật đúng trong ngữ cảnh và mô hình trích dẫn đúng điều luật đó.
- **Bịa nguồn:** trích số thứ tự không tồn tại trong ngữ cảnh.
- **Từ chối đúng:** ngữ cảnh không chứa điều luật đúng và mô hình trả lời bằng câu từ chối.

### Cổng từ chối ở tầng hệ thống (reranker gate)

Nếu điểm reranker (bge-reranker-v2-m3) của điều luật tốt nhất trong ngữ cảnh thấp hơn ngưỡng τ, hệ thống trả lời từ chối
ngay mà không gọi LLM (`REFUSAL_THRESHOLD`, [scripts/tune_refusal_gate.py](scripts/tune_refusal_gate.py)).
τ được chọn trên **300 câu validation lấy từ tập train nhưng không dùng cho SFT** (148 trả lời được, 152 không),
rồi áp dụng lên đúng các câu trả lời đã sinh ở bảng trên. Không sinh lại, nên mọi mô hình được so trên cùng quyết định của cổng.

| Ngưỡng τ (cách chọn) | Cổng giữ lại câu trả lời được | Cổng chặn câu không trả lời được | LoRA v1: trích đúng gold | LoRA v1: từ chối đúng | LoRA v1: faithfulness |
|---|---|---|---|---|---|
| không có cổng | 100% | 0% | 82.1% | 14.8% | 59.1% |
| 0.14 (giữ ≥ 95% trên val) | 96.7% | 7.4% | 82.1% | 18.5% | 59.6% |
| 0.32 (giữ ≥ 90%) | 94.3% | 11.1% | 81.3% | 22.2% | 60.5% |
| **0.67 (giữ ≥ 85%)** | **91.1%** | **33.3%** | **79.7%** | **40.7%** | **63.2%** |
| 0.99 (balanced accuracy tốt nhất) | 63.4% | 70.4% | 60.2% | 70.4% | 79.1% |

(Tỉ lệ giữ/chặn ở bảng là trên tập test; kết quả validation tương tự, ví dụ τ = 0.67 giữ 85.1% và chặn 27.6%.)

**Mọi mô hình với cổng τ = 0.67** (trước → sau khi có cổng):

| Mô hình | Trích đúng gold | Độ chính xác trích dẫn | **Từ chối đúng** | Faithfulness |
|---|---|---|---|---|
| Base | 41.5% → 40.6% | 61.9% → 63.1% | 51.9% → 59.3% | 46.2% → 46.6% |
| LoRA v1 | 82.1% → 79.7% | 68.5% → 72.7% | 14.8% → 40.7% | 59.1% → 63.2% |
| **LoRA v2 + cổng (cấu hình chọn)** | 78.0% → **76.4%** | 72.8% → **73.5%** | 37.0% → **44.4%** | 62.6% → **63.9%** |
| Teacher | 76.4% → 74.0% | 83.5% → 83.9% | 77.8% → 81.5% | 97.0% → 96.9% |

- Cổng giúp LoRA v1 nhiều (+25.9 điểm từ chối) nhưng giúp LoRA v2 ít hơn (+7.4 điểm): hai cách sửa **chặn phần lớn cùng một nhóm câu**,
  tức những câu mà điều luật trong ngữ cảnh rõ ràng không liên quan. Nhóm còn lại là hard negatives cùng chủ đề, cả hai cách đều chưa xử lý được.
- Cấu hình dùng cho demo: **LoRA v2 + cổng τ = 0.67**, có tỉ lệ từ chối, độ chính xác trích dẫn và faithfulness cao nhất trong các mô hình 3B,
  trong khi tỉ lệ trích đúng gold vẫn ngang teacher. Với miền pháp luật, từ chối khi thiếu căn cứ được ưu tiên hơn trả lời nhiều.
- Chỉ có 27 câu "nên từ chối", nên chênh lệch 1 câu bằng 3.7 điểm phần trăm; 40.7% và 44.4% chỉ khác nhau 1 câu.

**Nhận xét:**
- Ở τ = 0.67, LoRA v1 từ chối đúng tăng từ 14.8% lên 40.7%, độ chính xác trích dẫn từ 68.5% lên 72.7%, faithfulness từ 59.1% lên 63.2%,
  đổi lại tỉ lệ trích đúng gold chỉ giảm 2.4 điểm. Đây là ngưỡng đề xuất khi chạy demo.
- **Điểm reranker cao nhất chỉ là tín hiệu yếu** để biết ngữ cảnh có trả lời được câu hỏi hay không: balanced accuracy tốt nhất
  trên validation chỉ 63%. Lý do là mẫu "không trả lời được" dùng **hard negatives cùng chủ đề** (BM25 top-10), reranker vẫn chấm
  chúng là "liên quan" dù không chứa câu trả lời. Với câu hỏi ngoài phạm vi thực sự (không có điều luật nào cùng chủ đề),
  cổng dự kiến hiệu quả hơn nhiều; cần thêm tập test loại này để đo.
- Muốn chặn tốt hơn cần tín hiệu "điều luật này có trả lời câu hỏi không" thay vì "có liên quan không": ví dụ fine-tune reranker
  trên cặp (câu hỏi, điều luật gold / hard negative) của tập train, hoặc kết hợp cổng với LoRA v2.

**Hướng cải thiện tiếp theo:**
- Fine-tune reranker trên khoảng 2.4K cặp câu hỏi và điều luật gold của tập train, với hard negatives từ BM25; vừa tăng Recall@1
  của retrieval, vừa cho cổng từ chối một tín hiệu mạnh hơn.
- Áp dụng cổng lên LoRA v2 (đã học từ chối tốt hơn) và đo trên câu hỏi ngoài phạm vi.

## Cấu trúc

```
src/legalrag/
  data.py        đọc dataset, chunk điều luật dài theo khoản (có overlap), định dạng trích dẫn
  text.py        chuẩn hoá Unicode NFC + tách âm tiết/bigram
  retrieval.py   BM25 (tự cài bằng sparse matrix), Dense (FAISS), Hybrid (RRF), Reranker
  metrics.py     Recall@k, MRR, nDCG
  prompts.py     prompt có ngữ cảnh đánh số, parse trích dẫn, câu từ chối cố định
  sampling.py    ngữ cảnh "gold + hard negatives" cho dữ liệu SFT và đánh giá
  gate.py        cổng từ chối theo điểm reranker (chọn ngưỡng trên validation)
  pipeline.py    retrieve → rerank → (cổng) → generate
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
- [x] **Tuần 3:** đánh giá base / LoRA / teacher; phân tích lỗi; ablation tỉ lệ mẫu từ chối (17% và 35%)
- [ ] **Tuần 4:** Docker, demo, README hoàn chỉnh, quay GIF demo, đưa lên CV

## Ghi chú kỹ thuật

- **Chunking:** một số điều luật dài hơn 100K ký tự. Nếu không chunk, embedding model sẽ âm thầm cắt mất phần sau. Điểm của một điều luật là điểm của chunk tốt nhất (max-pooling), và chunk đó được đưa vào reranker/LLM.
- **RRF thay vì cộng điểm:** điểm BM25 không bị chặn còn cosine nằm trong [-1, 1]. RRF chỉ dùng thứ hạng nên không cần hiệu chỉnh thang điểm.
- **Đánh giá oracle và pipeline:** ở chế độ oracle, mọi mô hình nhận cùng ngữ cảnh (seed theo câu hỏi), nên so sánh base với LoRA không bị nhiễu bởi retrieval. Chế độ pipeline đo hệ thống end-to-end.
- **Hạn chế:** faithfulness chấm bằng LLM-as-judge nên có sai số. Nên kiểm tra tay một mẫu nhỏ để xem judge có đồng thuận với người chấm không.
