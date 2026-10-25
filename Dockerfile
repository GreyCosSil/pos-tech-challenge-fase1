# =============================================================================
# Dockerfile da API Telco Churn (FastAPI + JWT)
# =============================================================================

# Python 3.12 slim (o projeto exige >=3.11,<3.13)
FROM python:3.12-slim

WORKDIR /app

# Copiar dependencias primeiro (aproveita cache do Docker)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Mantem a estrutura do repositorio: src/app/main.py procura o modelo em ../../models
COPY src/app/ ./src/app/
COPY models/telco_churn_best_model.joblib models/telco_churn_experiment.json ./models/

# Roda como usuario sem privilegios
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Render injeta a env PORT. Localmente cai no default 8000.
# Segredos (JWT_SECRET, ADMIN_PASSWORD, USER_PASSWORD) vem do ambiente, nunca da imagem.
CMD ["sh", "-c", "uvicorn app.main:app --app-dir src --host 0.0.0.0 --port ${PORT:-8000}"]
