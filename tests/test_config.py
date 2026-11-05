import os
import tempfile
from src.config import deep_update, load_cfg


def test_deep_update():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    new = {"b": {"d": 4, "e": 5}, "f": 6}
    merged = deep_update(base, new)

    assert merged["a"] == 1
    assert merged["b"]["c"] == 2
    assert merged["b"]["d"] == 4
    assert merged["b"]["e"] == 5
    assert merged["f"] == 6


def test_load_base_cfg():
    cfg = load_cfg("configs/base.yaml")
    assert cfg["model"]["provider"] == "groq"
    assert cfg["model"]["name"] == "openai/gpt-oss-20b"
    assert cfg["retrieval"]["embedder"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg["retrieval"]["k"] == 3
    assert cfg["erag"]["theta_fast"] == 0.8
    assert cfg["erag"]["theta_gap"] == 0.6
    assert cfg["erag"]["theta_abs"] == 0.4


def test_override_loading():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("run:\n  system: erag\n  name: override_test\n")
        tmp_name = f.name

    try:
        cfg = load_cfg("configs/base.yaml", tmp_name)
        assert cfg["run"]["system"] == "erag"
        assert cfg["run"]["name"] == "override_test"
        assert cfg["retrieval"]["k"] == 3  # Preserved from base
    finally:
        os.remove(tmp_name)

