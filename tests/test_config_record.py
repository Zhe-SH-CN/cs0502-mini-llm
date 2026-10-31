from minillm.config import ExpConfig, MODELS, TASKS
from minillm.record import Recorder


def test_tasks_and_models_registered():
    assert set(TASKS) == {"courseqa", "jsonout", "intent"}
    for k, v in MODELS.items():
        assert v[0] and v[1]


def test_config_roundtrip():
    c = ExpConfig(run_id="x", model_key="qwen2.5-0.5b", task="intent", lora_r=16)
    d = c.to_dict()
    c2 = ExpConfig.from_dict(d)
    assert c2.lora_r == 16 and c2.hf_id == MODELS["qwen2.5-0.5b"][0]


def test_recorder_writes_run_and_csv(tmp_path):
    rec = Recorder(tmp_path)
    rec.log_run("r1", {"model_key": "m", "task": "courseqa"}, {"overall": 0.5, "n": 2},
                [{"id": "a"}], env={"ts": "t"})
    rec.log_run("r2", {"model_key": "m", "task": "intent"}, {"accuracy": 0.7}, [], env={"ts": "t"})
    runs = rec.load_runs()
    assert [r["run_id"] for r in runs] == ["r1", "r2"]
    assert (tmp_path / "r1.json").exists()
    csv = (tmp_path / "summary.csv").read_text(encoding="utf-8")
    assert "m_overall" in csv and "r2" in csv
