from minillm import metrics as M


def test_normalize_strips_punct_and_case():
    assert M.normalize_zh("Hello, 世界！") == M.normalize_zh("hello 世界")


def test_exact_match():
    assert M.exact_match("队列", "队列！") == 1.0
    assert M.exact_match("队列", "栈") == 0.0


def test_token_f1_partial():
    f = M.token_f1("私钥签名", "用私钥签名")
    assert 0.5 < f < 1.0
    assert M.token_f1("abc", "xyz") == 0.0


def test_contains():
    assert M.contains_answer("答案是队列没错", "队列") == 1.0
    assert M.contains_answer("栈", "队列") == 0.0


def test_safe_json_loads_fence_and_noise():
    assert M.safe_json_loads('```json\n{"a": 1}\n```') == {"a": 1}
    assert M.safe_json_loads('好的，结果如下 {"a": 1} 希望有帮助') == {"a": 1}
    assert M.safe_json_loads("不是json") is None


def test_schema_valid():
    schema = {"fields": {"name": {"type": "str"}, "n": {"type": "int"},
                         "ok": {"type": "bool"}, "cat": {"type": "str", "enum": ["a", "b"]}}}
    ok, errs = M.schema_valid({"name": "x", "n": 3, "ok": True, "cat": "a"}, schema)
    assert ok and errs == []
    ok, errs = M.schema_valid({"name": "x", "n": True, "cat": "z"}, schema)
    assert not ok
    assert any(e.startswith("type:n") for e in errs)
    assert any(e.startswith("enum:cat") for e in errs)
    assert any(e.startswith("missing:ok") for e in errs)


def test_field_accuracy():
    assert M.field_accuracy({"a": 1, "b": 2}, {"a": 1, "b": 9}, ["a", "b"]) == 0.5
    assert M.field_accuracy("notdict", {"a": 1}, ["a"]) == 0.0


def test_accuracy_macro_f1():
    assert M.accuracy(["a", "b"], ["a", "b"]) == 1.0
    assert M.macro_f1(["a", "b", "a"], ["a", "b", "b"]) > 0.5
