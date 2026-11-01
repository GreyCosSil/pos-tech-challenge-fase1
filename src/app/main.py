"""
API Telco Churn com JWT - versao para produçao - modularizada
pronta para deploy em container/cloud
"""

import json
import os
import time
from pathlib import Path
from typing import Literal

import joblib
from fastapi import Depends, FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field

from app.auth import (
    TOKEN_EXPIRE_MINUTES,
    authenticate_user,
    create_token,
    get_current_user,
)
from app.logging_config import setup_logging
from app.metrics import (
    LOGIN_ATTEMPTS,
    MODEL_LOADED,
    PREDICTION_LATENCY,
    PREDICTIONS_TOTAL,
)
from app.middleware import LoggingMiddleware
from src.prediction import predict_customer

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
API_VERSION = "1.0.0"

logger = setup_logging(os.getenv("LOG_LEVEL", "INFO"))


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


SimNao = Literal["No", "Yes"]
ServicoInternet = Literal["No", "No internet service", "Yes"]

EXEMPLO_CLIENTE = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.7,
    "TotalCharges": 151.65,
}


class ChurnRequest(BaseModel):
    """Dados do cliente, com os mesmos nomes de colunas do dataset Telco"""

    model_config = ConfigDict(json_schema_extra={"example": EXEMPLO_CLIENTE})

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1] = Field(..., description="1 se o cliente é idoso")
    Partner: SimNao
    Dependents: SimNao
    tenure: int = Field(..., ge=0, description="Meses como cliente")
    PhoneService: SimNao
    MultipleLines: Literal["No", "No phone service", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: ServicoInternet
    OnlineBackup: ServicoInternet
    DeviceProtection: ServicoInternet
    TechSupport: ServicoInternet
    StreamingTV: ServicoInternet
    StreamingMovies: ServicoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: SimNao
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]
    MonthlyCharges: float = Field(..., ge=0, description="Cobrança mensal")
    TotalCharges: float | None = Field(
        None, ge=0, description="Cobrança total (pode ser nulo para clientes novos)"
    )


class ChurnResponse(BaseModel):
    sucesso: bool
    churn: bool
    classe: Literal["No", "Yes"]
    probabilidade_churn: float
    probabilidades: dict[str, float]
    usuario: str


ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", MODELS_DIR / "telco_churn_best_model.joblib")
)
EXPERIMENT_PATH = MODELS_DIR / "telco_churn_experiment.json"

CLASSES = {0: "No", 1: "Yes"}

modelo = None
experimento = {}

try:
    modelo = joblib.load(MODEL_PATH)
    logger.info(f"modelo carregado de: {MODEL_PATH}")
except FileNotFoundError:
    logger.warning("modelo nao encontrado | servico de predicao indisponivel")

try:
    experimento = json.loads(EXPERIMENT_PATH.read_text())
except FileNotFoundError:
    logger.warning("metadados do experimento nao encontrados")

FEATURE_COLUMNS = experimento.get("feature_columns") or list(ChurnRequest.model_fields)

MODELO_OK = modelo is not None
MODEL_LOADED.set(1 if MODELO_OK else 0)


app = FastAPI(
    title="API de predição de churn - Telco",
    description="API de predição de churn de clientes com logging e monitoramento",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    # redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@2/bundles/redoc.standalone.js",
)

# Adiciona Middleware de Logging (intercepta todas as requisicoes)
app.add_middleware(LoggingMiddleware)

# Instrumentacao Prometheus (adiciona metricas automaticas + endpoint /metrics)
# Demonstracao de arquitetura: Prometheus e Grafana seriam implementados na fase 2
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.get("/")
def home():
    """Informações da API"""
    return {
        "api": "Telco Churn Predictor",
        "versao": API_VERSION,
        "ambiente": ENVIRONMENT,
        "modelo_carregado": MODELO_OK,
        "docs": "/docs",
        "metrics": "/metrics",
    }


@app.get("/health")
def health():
    """
    Verifica se a API esta funcionando corretamente
    """
    return {
        "status": "healthy" if MODELO_OK else "unhealthy",
        "modelo_carregado": MODELO_OK,
        "ambiente": ENVIRONMENT,
    }


@app.get("/model-info")
def model_info():
    """Metadados do modelo servido (algoritmo, parametros e metricas de teste)"""
    return {
        "modelo_carregado": MODELO_OK,
        "modelo": experimento.get("best_model"),
        "parametros": experimento.get("best_params"),
        "metricas_teste": experimento.get("test_metrics"),
        "features": FEATURE_COLUMNS,
    }


@app.post("/login")
def login(credentials: LoginRequest):
    """
    faz o login e retorna o token JWT
    """
    user = authenticate_user(credentials.username, credentials.password)

    if not user:
        LOGIN_ATTEMPTS.labels(status="falha").inc()
        logger.warning(f"login falhou para: {credentials.username}")
        raise HTTPException(status_code=401, detail="usuário ou senha incorretos")
    LOGIN_ATTEMPTS.labels(status="sucesso").inc()
    token = create_token(credentials.username, user["role"])
    return TokenResponse(access_token=token, expires_in=TOKEN_EXPIRE_MINUTES * 60)


@app.get("/me", tags=["Auth"])
def get_me(current_user: dict = Depends(get_current_user)):
    """
    Retorna infos do usuario logado
    """
    return current_user


@app.post("/predict", response_model=ChurnResponse)
def predict(
    request: Request,
    payload: ChurnRequest,
    current_user: dict = Depends(get_current_user),
):
    """Faz predicao de churn (requer autenticacao)."""
    start = time.perf_counter()
    trace_id = request.state.trace_id

    if not MODELO_OK:
        raise HTTPException(status_code=503, detail="Modelo nao disponivel")

    prediction = predict_customer(modelo, payload.model_dump())
    pred_idx = prediction.prediction
    probs = [prediction.probability_no, prediction.probability_yes]
    classe = CLASSES[pred_idx]
    confidence = float(max(probs))

    latency = time.perf_counter() - start

    # Metricas Prometheus
    PREDICTIONS_TOTAL.labels(classe=classe, user=current_user["username"]).inc()
    PREDICTION_LATENCY.observe(latency)

    # Log estruturado
    logger.info(
        "prediction_completed",
        extra={
            "trace_id": trace_id,
            "user": current_user["username"],
            "classe": classe,
            "confidence": round(confidence, 4),
            "latency_ms": round(latency * 1000, 2),
            "features": payload.model_dump(),
        },
    )

    return ChurnResponse(
        sucesso=True,
        churn=pred_idx == 1,
        classe=classe,
        probabilidade_churn=round(float(probs[1]), 4),
        probabilidades={
            CLASSES[int(c)]: round(float(p), 4)
            for c, p in zip(modelo.classes_, probs)
        },
        usuario=current_user["username"],
    )
