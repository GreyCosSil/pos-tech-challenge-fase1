import pytest

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


async def test_middleware_headers_injection(api_client):
    """Valida se o middleware injeta os headers X-Trace-ID e X-Response-Time-Ms"""
    response = await api_client.get("/health")
    assert response.status_code == 200
    assert "X-Trace-ID" in response.headers
    assert "X-Response-Time-Ms" in response.headers


async def test_prometheus_metrics_endpoint(api_client):
    """Garante que o endpoint /metrics está ativo e expõe as métricas customizadas"""
    response = await api_client.get("/metrics")
    assert response.status_code == 200
    assert "churn_predictions_total" in response.text
    assert "model_loaded" in response.text


async def test_metrics_increment_on_login_failure(api_client):
    """Testa se a tentativa de login malsucedida é contabilizada"""
    login_payload = {"username": "usuario_invalido", "password": "senha_errada"}
    response = await api_client.post("/login", json=login_payload)
    assert response.status_code == 401

    metrics_response = await api_client.get("/metrics")
    assert 'login_attempts_total{status="falha"}' in metrics_response.text
