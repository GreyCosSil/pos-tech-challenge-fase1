import os

# credenciais fixas para os testes, independentes do .env local
# (definidas antes do import do app; load_dotenv nao sobrescreve variaveis existentes)
os.environ["JWT_SECRET"] = "segredo-de-teste"
os.environ["ADMIN_PASSWORD"] = "admin-teste"
os.environ["USER_PASSWORD"] = "usuario_teste"
