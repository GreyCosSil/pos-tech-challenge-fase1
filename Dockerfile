# =============================================================================
# Dockerfile da API Telco Churn (FastAPI + JWT)
# =============================================================================

# Python 3.12 slim (o projeto exige >=3.11,<3.13)
FROM python:3.12-slim

WORKDIR /app

# 1. Copia apenas os manifestos de dependência e README primeiro (aproveita cache do Docker)
COPY pyproject.toml README.md ./

# 2. Atualiza o pip e instala apenas as dependências de produção do pyproject.toml
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# 3. Mantém a estrutura do repositório
COPY src/app/ ./src/app/
COPY models/telco_churn_best_model.joblib models/telco_churn_experiment.json ./models/

# 4. Roda como usuário sem privilégios por segurança
RUN useradd --create-home appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Render injeta a env PORT. Localmente cai no default 8000.
# Segredos (JWT_SECRET, ADMIN_PASSWORD, USER_PASSWORD) vêm do ambiente, nunca da imagem.
CMD ["sh", "-c", "uvicorn app.main:app --app-dir src --host 0.0.0.0 --port ${PORT:-8000}"]