"""Gradio demo that calls the REST API.

    API_URL=http://localhost:8080 python app/ui.py
"""
from __future__ import annotations

import html
import os
import re

import gradio as gr
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8080")

EXAMPLES = [
    "Công an xã có được xử phạt lỗi không mang bằng lái xe không?",
    "Người lao động nghỉ việc cần báo trước bao nhiêu ngày?",
    "Không đội mũ bảo hiểm khi đi xe máy bị phạt bao nhiêu tiền?",
]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');

:root {
  --ink: #1c2430;
  --muted: #6b7482;
  --line: #e4e7ec;
  --surface: #ffffff;
  --canvas: #f6f7f9;
  --accent: #1f4e8c;
  --accent-soft: #e8eef7;
}
.dark {
  --ink: #e6e9ee;
  --muted: #9aa3b1;
  --line: #2b323d;
  --surface: #171b22;
  --canvas: #0f1217;
  --accent: #7fa7e0;
  --accent-soft: #1d2838;
}

html, body, gradio-app, .main, .gradio-container { background: var(--canvas) !important; }
.gradio-container, .gradio-container * { font-family: 'Be Vietnam Pro', system-ui, sans-serif !important; }
.gradio-container { max-width: 880px !important; margin: 0 auto !important; }
footer { display: none !important; }

.hero { padding: 28px 0 8px; }
.hero h1 { font-size: 26px; font-weight: 700; color: var(--ink); margin: 0 0 6px; letter-spacing: -0.01em; }
.hero p { color: var(--muted); font-size: 14.5px; margin: 0; line-height: 1.6; }

.card {
  background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
  padding: 20px 22px; color: var(--ink);
}
.label {
  font-size: 11.5px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 10px;
}
.answer { font-size: 15.5px; line-height: 1.75; }
.answer.refused { color: var(--muted); font-style: italic; }
.meta { margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--line);
        font-size: 12px; color: var(--muted); display: flex; gap: 16px; flex-wrap: wrap; }

.cite {
  display: inline-flex; align-items: center; justify-content: center;
  min-width: 18px; height: 18px; padding: 0 5px; margin: 0 1px;
  border-radius: 5px; background: var(--accent-soft); color: var(--accent) !important;
  font-size: 11px; font-weight: 600; text-decoration: none !important; vertical-align: 2px;
}
.cite:hover { background: var(--accent); color: #fff !important; }

.source { display: flex; gap: 14px; padding: 16px 0; border-top: 1px solid var(--line); }
.source:first-of-type { border-top: none; padding-top: 4px; }
.badge {
  flex: none; width: 26px; height: 26px; border-radius: 7px;
  display: flex; align-items: center; justify-content: center;
  font-size: 12.5px; font-weight: 600;
  border: 1.5px solid var(--line); color: var(--muted);
}
.source.cited .badge { background: var(--accent); border-color: var(--accent); color: #fff; }
.source-head { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; margin-bottom: 6px; }
.source-title { font-weight: 600; font-size: 14px; color: var(--ink); }
.tag { font-size: 11px; font-weight: 500; padding: 2px 8px; border-radius: 999px;
       color: var(--muted); background: var(--canvas); border: 1px solid var(--line); }
.source.cited .tag { color: var(--accent); background: var(--accent-soft); border-color: transparent; }
.source-text { font-size: 13.5px; line-height: 1.65; color: var(--muted); white-space: pre-line; }
.source details summary { cursor: pointer; font-size: 12.5px; color: var(--accent); margin-top: 6px; }

.empty { text-align: center; color: var(--muted); font-size: 14px; padding: 36px 12px; }
.error { border-color: #e5b4b4; color: #9b2c2c; }

#ask-btn { background: var(--accent) !important; color: #fff !important; border: none !important; font-weight: 600 !important; }
"""

EMPTY = '<div class="card empty">Nhập câu hỏi để bắt đầu. Câu trả lời sẽ kèm trích dẫn tới điều luật cụ thể.</div>'


def _render_answer(answer: str, n_docs: int) -> str:
    text = html.escape(answer)

    def link(match: re.Match) -> str:
        idx = int(match.group(1))
        if 1 <= idx <= n_docs:
            return f'<a class="cite" href="#src-{idx}">{idx}</a>'
        return match.group(0)

    return re.sub(r"\[(\d+)\]", link, text).replace("\n", "<br>")


def _render_source(i: int, doc: dict, cited: bool) -> str:
    title, _, body = doc["text"].partition("\n")
    body = body.strip()
    preview, rest = body[:420], body[420:]
    text = html.escape(preview)
    if rest:
        text += f'…<details><summary>Xem toàn văn</summary>{html.escape(body)}</details>'
    tag = "Được trích dẫn" if cited else "Tham khảo"
    return f"""
    <div class="source{' cited' if cited else ''}" id="src-{i}">
      <div class="badge">{i}</div>
      <div>
        <div class="source-head">
          <span class="source-title">{html.escape(doc['citation'])}</span>
          <span class="tag">{tag}</span>
        </div>
        <div class="source-text"><b>{html.escape(title)}</b>\n{text}</div>
      </div>
    </div>"""


def ask(question: str) -> tuple[str, str]:
    if not question or not question.strip():
        return EMPTY, ""
    try:
        response = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=180)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return f'<div class="card error">Không gọi được API ({html.escape(str(exc))}).</div>', ""

    docs = data["contexts"]
    cited = {c["index"] for c in data["citations"]}
    t = data["timings_ms"]
    meta = " ".join(
        f"<span>{label}: {t[key] / 1000:.2f}s</span>"
        for key, label in (("retrieve", "Truy xuất"), ("rerank", "Xếp hạng lại"), ("generate", "Sinh câu trả lời"))
        if key in t
    )
    answer_html = f"""
    <div class="card">
      <div class="label">Trả lời</div>
      <div class="answer{' refused' if data['refused'] else ''}">{_render_answer(data['answer'], len(docs))}</div>
      <div class="meta">{meta}</div>
    </div>"""

    sources_html = ""
    if docs:
        items = "".join(_render_source(i, d, i in cited) for i, d in enumerate(docs, start=1))
        sources_html = f'<div class="card"><div class="label">Căn cứ pháp lý ({len(docs)})</div>{items}</div>'
    return answer_html, sources_html


THEME = gr.themes.Base(primary_hue="blue", neutral_hue="slate", radius_size="md")
# Gradio 6 takes theme/css in launch(); Gradio 4-5 only accept them in Blocks().
_GRADIO6 = int(gr.__version__.split(".")[0]) >= 6
STYLE = {"theme": THEME, "css": CSS}


def build_demo(initial_question: str = "") -> gr.Blocks:
    with gr.Blocks(title="Trợ lý pháp luật", **({} if _GRADIO6 else STYLE)) as demo:
        gr.HTML(
            '<div class="hero"><h1>Trợ lý hỏi đáp pháp luật</h1>'
            "<p>Tra cứu trên hơn 61.000 điều luật Việt Nam. Mỗi câu trả lời chỉ dựa trên các điều luật "
            "được truy xuất và ghi rõ nguồn.</p></div>"
        )
        with gr.Group():
            question = gr.Textbox(initial_question, show_label=False, lines=2, max_lines=6,
                                  placeholder="Ví dụ: Nghỉ việc cần báo trước bao nhiêu ngày?")
        with gr.Row():
            gr.Examples(EXAMPLES, inputs=question, label="Câu hỏi mẫu")
        submit = gr.Button("Hỏi", elem_id="ask-btn", size="lg")
        answer = gr.HTML(EMPTY)
        sources = gr.HTML()

        submit.click(ask, inputs=question, outputs=[answer, sources])
        question.submit(ask, inputs=question, outputs=[answer, sources])
        if initial_question:
            demo.load(ask, inputs=question, outputs=[answer, sources])
    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
                        **(STYLE if _GRADIO6 else {}))
