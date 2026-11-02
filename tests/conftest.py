"""
Configurações globais e fixtures do Pytest (tests/conftest.py)
"""

import os
import sys
from pathlib import Path

import httpx
import pytest

# 1. Adiciona o diretório 'src' ao sys.path para garantir que os imports funcionem
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# 2. Injeta variáveis de ambiente padrão para o ambiente de testes
os.environ["JWT_SECRET"] = "segredo_para_testes_locais_123"
os.environ["ADMIN_PASSWORD"] = "admin_password"
os.environ["USER_PASSWORD"] = "user_password"
os.environ["ENVIRONMENT"] = "testing"


@pytest.fixture
async def api_client(anyio_backend):
    from app.main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        yield client