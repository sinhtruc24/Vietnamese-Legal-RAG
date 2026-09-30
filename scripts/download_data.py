"""Download the Zalo AI 2021 Legal Text Retrieval dataset from the Hugging Face Hub.

Usage: python scripts/download_data.py --out data/zalo_legal
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from legalrag.data import LegalDataset

REPO_ID = "GreenNode/zalo-ai-legal-text-retrieval-vn"
FILES = ["corpus.jsonl", "queries.jsonl", "qrels/train.jsonl", "qrels/test.jsonl"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/zalo_legal"))
    args = parser.parse_args()

    for name in FILES:
        target = args.out / name
        if target.exists():
            print(f"skip {target} (exists)")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(REPO_ID, name, repo_type="dataset")
        shutil.copy(cached, target)
        print(f"saved {target}")

    ds = LegalDataset.load(args.out)
    print(f"\narticles: {len(ds.corpus):,}  queries: {len(ds.queries):,}")
    for split in ds.qrels_by_split:
        print(f"{split:>5}: {len(ds.qrels(split)):,} queries with gold articles")


if __name__ == "__main__":
    main()
