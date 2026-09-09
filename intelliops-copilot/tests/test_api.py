import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["baseline_loaded"] is True
    assert data["variant_loaded"] is True


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "baseline" in data
    assert "variant" in data

    baseline = data["baseline"]
    assert "param_count" in baseline and isinstance(baseline["param_count"], int)
    assert "vocab_size" in baseline and isinstance(baseline["vocab_size"], int)
    assert "val_perplexity" in baseline and isinstance(baseline["val_perplexity"], (int, float))

    variant = data["variant"]
    assert "param_count" in variant and isinstance(variant["param_count"], int)
    assert "vocab_size" in variant and isinstance(variant["vocab_size"], int)
    assert "val_perplexity" in variant and isinstance(variant["val_perplexity"], (int, float))


def test_generate_endpoint_default():
    payload = {
        "prompt": "Hello",
        "max_new_tokens": 10,
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prompt"] == "Hello"
    assert data["model"] == "baseline"
    assert "generated_text" in data
    assert len(data["generated_text"]) >= len("Hello")


def test_generate_endpoint_baseline_and_variant():
    prompt = "user: How do I deploy a model?\nassistant:"

    res_base = client.post("/generate", json={"prompt": prompt, "model": "baseline", "max_new_tokens": 20, "seed": 42})
    assert res_base.status_code == 200
    data_base = res_base.json()
    assert data_base["model"] == "baseline"

    res_var = client.post("/generate", json={"prompt": prompt, "model": "variant", "max_new_tokens": 20, "seed": 42})
    assert res_var.status_code == 200
    data_var = res_var.json()
    assert data_var["model"] == "variant"

    assert data_base["generated_text"] != data_var["generated_text"]


def test_generate_endpoint_invalid_model():
    res = client.post("/generate", json={"prompt": "Test", "model": "non_existent_model"})
    assert res.status_code == 400
    assert "Invalid model" in res.json()["detail"]


def test_generate_instruction_endpoint():
    payload = {
        "instruction": "write a short monologue about courage",
        "max_new_tokens": 30,
        "seed": 42,
    }
    response = client.post("/generate-instruction", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["instruction"] == payload["instruction"]
    assert "generated_response" in data
    assert "model" in data

