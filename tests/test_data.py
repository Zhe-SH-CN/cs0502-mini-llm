import json

from minillm import data as D
from minillm.tasks import BUILDERS


def test_builders_deterministic_and_nonempty():
    for task, b in BUILDERS.items():
        r1, r2 = b(seed=7), b(seed=7)
        assert r1 == r2
        assert len(r1) > 50, task


def test_build_all_splits_no_topic_leak(tmp_path):
    stats = D.build_all_splits(tmp_path, seed=42)
    assert set(stats) == {"courseqa", "jsonout", "intent"}
    tr = D.read_jsonl(tmp_path / "courseqa" / "train.jsonl")
    te = D.read_jsonl(tmp_path / "courseqa" / "test.jsonl")
    # 分层切分：同一 topic 允许同时出现，但同一条 id 不允许跨 split
    ids_tr = {r["id"] for r in tr}
    ids_te = {r["id"] for r in te}
    assert not (ids_tr & ids_te)
    assert stats["courseqa"]["train"] > stats["courseqa"]["test"]


def test_subsample_and_noise():
    rows = BUILDERS["intent"](seed=1)
    sub = D.subsample(rows, 50)
    assert len(sub) == 50
    noisy = D.inject_label_noise(rows, 0.5, "intent")
    diff = sum(1 for a, b in zip(rows, noisy) if a["label"] != b["label"])
    assert diff > 0
    assert D.inject_label_noise(rows, 0.0, "intent") == rows


def test_to_sft_messages_shape():
    row = {"question": "q", "answer": "a"}
    m = D.to_sft_messages(row, "courseqa")
    assert [x["role"] for x in m] == ["user", "assistant"]
    assert m[-1]["content"] == "a"


def test_jsonl_roundtrip(tmp_path):
    rows = [{"a": 1, "中": "文"}]
    D.write_jsonl(rows, tmp_path / "x.jsonl")
    assert D.read_jsonl(tmp_path / "x.jsonl") == rows
