from fastapi.testclient import TestClient

from app.main import EXEMPLO_CLIENTE, app

client = TestClient(app)


def _token():
    response = client.post(
        "/login", json={"username": "usuario_teste", "password": "usuario_teste"}
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def _headers():
    return {"Authorization": f"Bearer {_token()}"}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["modelo_carregado"] is True


def test_login_invalido():
    response = client.post(
        "/login", json={"username": "usuario_teste", "password": "errada"}
    )
    assert response.status_code == 401


def test_predict_sem_token():
    response = client.post("/predict", json=EXEMPLO_CLIENTE)
    assert response.status_code == 403


def test_predict_cliente_alto_risco():
    response = client.post("/predict", json=EXEMPLO_CLIENTE, headers=_headers())
    assert response.status_code == 200
    body = response.json()
    assert body["churn"] is True
    assert body["classe"] == "Yes"
    assert abs(sum(body["probabilidades"].values()) - 1) < 1e-3
    assert body["usuario"] == "usuario_teste"


def test_predict_cliente_baixo_risco():
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
    response = client.post("/predict", json=cliente, headers=_headers())
    assert response.status_code == 200
    assert response.json()["churn"] is False


def test_predict_total_charges_nulo():
    cliente = {**EXEMPLO_CLIENTE, "tenure": 0, "TotalCharges": None}
    response = client.post("/predict", json=cliente, headers=_headers())
    assert response.status_code == 200


def test_predict_categoria_invalida():
    cliente = {**EXEMPLO_CLIENTE, "Contract": "Three year"}
    response = client.post("/predict", json=cliente, headers=_headers())
    assert response.status_code == 422
