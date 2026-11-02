"""Modulo de Autenticacao JWT
Funcoes para criar e validar tokens JWT"""

import hmac
import os
from datetime import datetime, timedelta
from pathlib import Path

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

# carrega o .env da raiz do projeto; variaveis ja definidas no ambiente
# (container/cloud) tem prioridade sobre o arquivo
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _required_env(name: str) -> str:
    """Le uma variavel de ambiente obrigatoria, falhando na inicializacao se ausente"""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Variavel de ambiente obrigatoria nao definida: {name} (veja .env.example)"
        )
    return value


SECRET_KEY = _required_env("JWT_SECRET")
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = int(os.getenv("TOKEN_EXPIRE_MINUTES", "15"))

USERS_DB = {
    "admin": {"password": _required_env("ADMIN_PASSWORD"), "role": "admin"},
    "usuario_teste": {"password": _required_env("USER_PASSWORD"), "role": "user"},
}

security = HTTPBearer()


def create_token(username: str, role: str) -> str:
    """
    Cria um token JWT com expiração
    """
    expire = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)  # noqa: DTZ003
    payload = {
        "sub": username,
        "role": role,
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(security)) -> dict:
    """
    Extrai e avalia o user do token jwt
    """
    try:
        payload = jwt.decode(token.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return {"username": payload["sub"], "role": payload["role"]}
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401, detail="Token Expirado. Faça login de novo"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token Invalido")


def authenticate_user(username: str, password: str) -> dict | None:
    """Verifica credenciais do usuario"""
    user = USERS_DB.get(username)
    # compare_digest evita ataques de timing na comparacao de senha
    if user and hmac.compare_digest(user["password"], password):
        return {"username": username, "role": user["role"]}
    return None
