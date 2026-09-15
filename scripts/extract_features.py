from __future__ import annotations

import argparse
import re

import pandas as pd


CONFLICT_KEYWORDS = [
    "冲突", "矛盾", "占用", "违法", "违规", "重叠", "侵占", "纠纷", "争议", "压占",
    "生态红线", "永久基本农田", "建设用地", "耕地保护", "空间管制",
]
NON_CONFLICT_KEYWORDS = [
    "规划", "协调", "保护", "审批", "公示", "批复", "通知", "实施", "评估", "管理",
]
NEGATIVE_WORDS = ["不得", "禁止", "严禁", "违法", "违规", "未批", "占用", "破坏"]
GOVERNMENT_PHRASES = ["国务院", "自然资源部", "人民政府", "办公厅", "主管部门", "规划编制"]


def count_terms(text: str, terms: list[str]) -> int:
    return sum(text.count(term) for term in terms)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract keyword and text-statistical features from the unified dataset.")
    parser.add_argument("--input", required=True, help="Path to data/unified_land_use_dataset.csv")
    parser.add_argument("--output", required=True, help="Output feature CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    rows = []
    for _, row in df.iterrows():
        text = str(row["content"])
        word_count = max(int(row.get("word_count", len(text))), 1)
        conflict_count = count_terms(text, CONFLICT_KEYWORDS)
        non_conflict_count = count_terms(text, NON_CONFLICT_KEYWORDS)
        rows.append(
            {
                "sample_id": row["sample_id"],
                "conflict_keyword_count": conflict_count,
                "non_conflict_keyword_count": non_conflict_count,
                "keyword_density": conflict_count / word_count,
                "conflict_ratio": conflict_count / max(conflict_count + non_conflict_count, 1),
                "negative_word_count": count_terms(text, NEGATIVE_WORDS),
                "question_mark_count": text.count("?") + text.count("？"),
                "exclamation_mark_count": text.count("!") + text.count("！"),
                "government_phrase_count": count_terms(text, GOVERNMENT_PHRASES),
                "word_count": word_count,
                "label": int(row["label"]),
            }
        )
    pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
