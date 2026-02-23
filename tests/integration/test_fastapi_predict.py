def test_health_endpoint(fastapi_client):
    """Validate that health endpoint behaves as expected.

    This test guards against regressions in the associated workflow or API behavior.
    """
    response = fastapi_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_success(fastapi_client):
    """Validate that predict endpoint success behaves as expected.

    This test guards against regressions in the associated workflow or API behavior.
    """
    payload = {"rows": [{"age": 45, "combined_text": "test incident text"}]}
    response = fastapi_client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predictions" in body
    assert "prediction_probabilities" in body
    assert len(body["predictions"]) == 1
    assert len(body["prediction_probabilities"]) == 1


def test_predict_endpoint_signature_error(fastapi_client):
    """Validate that predict endpoint signature error behaves as expected.

    This test guards against regressions in the associated workflow or API behavior.
    """
    payload = {"rows": [{"age": 45}]}
    response = fastapi_client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Missing required feature columns" in response.json()["detail"]
