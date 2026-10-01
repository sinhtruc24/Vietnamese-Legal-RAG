import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from legalrag.data import read_jsonl, write_jsonl
from legalrag.evaluation import judge_faithfulness, score, summarize
from legalrag.generator import QuotaExhausted
from legalrag.prompts import REFUSAL

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(qid, answer, gold, n_docs=3):
    return score({"id": qid, "gold": gold, "n_docs": n_docs, "context": "ctx", "answer": answer, "latency_ms": 10.0})


def test_score_and_summarize():
    rows = [
        _row("a", "Đúng [1].", [1]),          # answerable, cites gold
        _row("b", "Sai nguồn [2] [9].", [1]),  # answerable, wrong + invalid citation
        _row("c", REFUSAL, []),                # negative, correctly refused
        _row("d", "Bịa [1].", []),             # negative, should have refused
    ]
    s = summarize(rows)
    assert s["n_answerable"] == 2 and s["n_negative"] == 2
    assert s["gold_cite_rate"] == 0.5
    assert s["refusal_acc"] == 0.5
    assert s["invalid_cite_rate"] == round(1 / 3, 4)
    assert s["faithfulness"] is None and s["n_judged"] == 0


class _Judge:
    def __init__(self, replies):
        self.replies = list(replies)

    def generate(self, messages):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_judge_parses_verdicts():
    judge = _Judge(['```json\n{"faithful": true}\n```', '{"faithful": false, "reason": "x"}', "không phải json"])
    assert judge_faithfulness(judge, "c", "a") is True
    assert judge_faithfulness(judge, "c", "a") is False
    assert judge_faithfulness(judge, "c", "a") is None


def test_judge_results_resumes_after_quota(tmp_path):
    judge_results = _load_script("judge_results")
    path = tmp_path / "gen_lora.jsonl"
    write_jsonl(path, [_row("a", "x [1]", [1]), _row("b", "y [1]", [1]), _row("c", REFUSAL, [])])
    args = SimpleNamespace(workers=1, judge_model="m")

    # Day 1: one verdict, then the daily quota runs out.
    assert judge_results.judge_file(path, _Judge(['{"faithful": true}', QuotaExhausted("PerDay")]), args) is False
    rows = list(read_jsonl(path))
    assert sum(r.get("faithful") is not None for r in rows) == 1

    # Day 2: only the remaining answered row is judged; the refusal is never sent.
    assert judge_results.judge_file(path, _Judge(['{"faithful": false}']), args) is True
    rows = list(read_jsonl(path))
    assert [r.get("faithful") for r in rows] == [True, False, None]
    summary = json.loads((tmp_path / "gen_lora_summary.json").read_text(encoding="utf-8"))
    assert summary["faithfulness"] == 0.5 and summary["n_judged"] == 2 and summary["judge_model"] == "m"
