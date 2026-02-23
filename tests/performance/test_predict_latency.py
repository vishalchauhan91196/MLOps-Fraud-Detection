import time


def test_predict_latency_under_reasonable_limit(fastapi_client):
    """Validate that predict latency under reasonable limit behaves as expected.

    This test guards against regressions in the associated workflow or API behavior.
    """
    payload = {"rows": [{"age": 45, "combined_text": "quick latency test"} for _ in range(20)]}
    start = time.perf_counter()
    response = fastapi_client.post("/predict", json=payload)
    elapsed = time.perf_counter() - start
    assert response.status_code == 200
    assert elapsed < 1.0
