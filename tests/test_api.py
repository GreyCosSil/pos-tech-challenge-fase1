import pytest

from app.main import EXEMPLO_CLIENTE

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


async def _token(client):
    response = await client.post(
        "/login", json={"username": "usuario_teste", "password": "user_password"}
    )
    assert response.status_code == 200
    return response.json()["access_token"]


async def _headers(client):
    token = await _token(client)
    return {"Authorization": f"Bearer {token}"}


async def test_health(api_client):
    response = await api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["modelo_carregado"] is True


async def test_login_invalido(api_client):
    response = await api_client.post(
        "/login", json={"username": "usuario_teste", "password": "errada"}
    )
    assert response.status_code == 401


async def test_predict_sem_token(api_client):
    response = await api_client.post("/predict", json=EXEMPLO_CLIENTE)
    assert response.status_code in {401, 403}


async def test_predict_cliente_alto_risco(api_client):
    headers = await _headers(api_client)
    response = await api_client.post(
        "/predict", json=EXEMPLO_CLIENTE, headers=headers
    )
    assert response.status_code == 200
    body = response.json()
    assert body["churn"] is True
    assert body["classe"] == "Yes"
    assert abs(sum(body["probabilidades"].values()) - 1) < 1e-3
    assert body["usuario"] == "usuario_teste"


async def test_predict_cliente_baixo_risco(api_client):
    cliente = {
        **EXEMPLO_CLIENTE,
        "tenure": 70,
        "Contract": "Two year",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "TechSupport": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 55.0,
        "TotalCharges": 3850.0,
    }
    headers = await _headers(api_client)
    response = await api_client.post("/predict", json=cliente, headers=headers)
    assert response.status_code == 200
    assert response.json()["churn"] is False


async def test_predict_total_charges_nulo(api_client):
    cliente = {**EXEMPLO_CLIENTE, "tenure": 0, "TotalCharges": None}
    headers = await _headers(api_client)
    response = await api_client.post("/predict", json=cliente, headers=headers)
    assert response.status_code == 200


async def test_predict_categoria_invalida(api_client):
    cliente = {**EXEMPLO_CLIENTE, "Contract": "Three year"}
    headers = await _headers(api_client)
    response = await api_client.post("/predict", json=cliente, headers=headers)
    assert response.status_code == 422
