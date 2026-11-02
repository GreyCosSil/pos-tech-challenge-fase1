"""
Testes unitários para o Middleware de Logging e Métricas Prometheus
"""

import httpx
import pytest

from app.main import app

pytestmark = pytest.mark.integration

# Transport ASGI nativo do httpx (funciona com qualquer versão do httpx)
transport = httpx.ASGITransport(app=app)
client = httpx.Client(transport=transport, base_url="http://testserver")


def test_middleware_headers_injection():
    """Valida se o middleware injeta os headers X-Trace-ID e X-Response-Time-Ms"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Trace-ID" in response.headers
    assert "X-Response-Time-Ms" in response.headers


def test_prometheus_metrics_endpoint():
    """Garante que o endpoint /metrics está ativo e expõe as métricas customizadas"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "churn_predictions_total" in response.text
    assert "model_loaded" in response.text


def test_metrics_increment_on_login_failure():
    """Testa se a tentativa de login malsucedida é contabilizada"""
    login_payload = {"username": "usuario_invalido", "password": "senha_errada"}
    response = client.post("/login", json=login_payload)
    assert response.status_code == 401

    metrics_response = client.get("/metrics")
    assert 'login_attempts_total{status="falha"}' in metrics_response.text