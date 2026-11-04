# 🤖 API de previsão de churn para telecomunicações

API FastAPI que usa um modelo treinado para prever churn de clientes. Este guia
explica como configurar o ambiente, iniciar a API e executar os testes integrados.

## 📁 Estrutura do Repositório

* `src/`: Código fonte para pré-processamento, treinamento e predição.
* `notebooks/`: Análises exploratórias e protótipos de dados.
* `data/`: Diretório para armazenar os datasets localmente (ignorado pelo Git).
* `models/`: Diretório para os modelos treinados (ignorado pelo Git).
* `tests/`: Testes unitários do código.
* `docs/`: Documentação adicional.

## 🚀 Como Configurar (Setup)

## Pré-requisitos

- Python 3.11 ou 3.12 para executar a aplicação e os testes localmente.
- Docker Desktop ou Docker Engine, opcionalmente, para iniciar a API em um
  container.

## 1. Obter o projeto

```bash
git clone https://github.com/GreyCosSil/pos-tech-challenge-fase1.git
cd pos-tech-challenge-fase1
```

## 2. Configurar as variáveis de ambiente

Crie o arquivo local `.env` a partir do exemplo:

```bash
cp .env.example .env
```

No PowerShell:

```powershell
Copy-Item .env.example .env
```

Edite `.env` e preencha as três variáveis obrigatórias. Gere um segredo aleatório
com Python:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Copie o valor gerado para `JWT_SECRET`. O comando gera 32 bytes (256 bits).
Defina senhas locais para `ADMIN_PASSWORD` e `USER_PASSWORD`. Exemplo de formato:

```dotenv
JWT_SECRET=cole_aqui_o_segredo_gerado
ADMIN_PASSWORD=defina_uma_senha_para_admin
USER_PASSWORD=defina_uma_senha_para_usuario_teste
TOKEN_EXPIRE_MINUTES=15
ENVIRONMENT=development
LOG_LEVEL=INFO
```

As três primeiras variáveis são obrigatórias; as demais são opcionais. Não
compartilhe nem faça commit do `.env`, pois ele contém segredos.

## 3. Preparar o ambiente Python

Crie um ambiente virtual com Python 3.11 ou 3.12 e ative-o.

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instale o projeto e as dependências de desenvolvimento, incluindo pytest:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 4. Iniciar a API localmente

Com o ambiente virtual ativado e o `.env` preenchido, execute na raiz do projeto:

```bash
python -m uvicorn app.main:app --app-dir src --host 127.0.0.1 --port 8000
```

Abra <http://localhost:8000/docs> para a documentação interativa, ou
<http://localhost:8000/health> para verificar a saúde da API. Para encerrar,
pressione `Ctrl+C`.

## 5. Executar os testes unitários e integrados

Em outro terminal, entre na pasta do projeto e ative o mesmo ambiente virtual.
Os testes são classificados com os marcadores `unit` e `integration`:

- **Unitários**: validam módulos ou funções isoladamente, usando dependências
  simuladas quando necessário (`tests/test_data.py`).
- **Integrados**: exercitam a aplicação FastAPI por requisições HTTP ASGI,
  verificando endpoints, autenticação, middleware e métricas (`tests/test_api.py`
  e `tests/test_metrics_and_middleware.py`).

Execute toda a suíte:

```bash
python -m pytest -q
```

Para executar somente os testes unitários:

```bash
python -m pytest -q -m unit
```

Para executar somente os testes integrados:

```bash
python -m pytest -q -m integration
```

Também é possível executar os testes dos endpoints da API:

```bash
python -m pytest -q tests/test_api.py
```

Para executar um caso individual:

```bash
python -m pytest -q tests/test_api.py::test_health
```

Substitua `test_health` pelo nome de outro teste definido nos arquivos de
`tests/`. O pytest retorna código diferente de zero quando encontra uma falha.

## Alternativa: iniciar a API com Docker

Com Docker instalado e em execução, construa a imagem na raiz do projeto:

```bash
docker build -t telco-churn-api .
```

Inicie o container usando as variáveis definidas no `.env`:

```bash
docker run --rm --env-file .env -p 8000:8000 telco-churn-api
```

A API ficará disponível em <http://localhost:8000/docs>. Encerre com `Ctrl+C`.
Essa imagem contém as dependências de produção para servir a API; para executar
os testes, use o ambiente Python de desenvolvimento descrito acima.


## API em produção no Render

- API: <https://api-telco-churn-prod-93yo.onrender.com>
- Documentação interativa: <https://api-telco-churn-prod-93yo.onrender.com/docs>
- Saúde da API: <https://api-telco-churn-prod-93yo.onrender.com/health>
- Frontend: <https://frontend-telco-churn-93yo.onrender.com>

## 👥 Integrantes do Grupo e Responsabilidades


| Integrante | Principais Contribuições |
| :--- | :--- |
| **Greyce** | Configuração de esteira CI/CD, gerenciamento de ambientes e infraestrutura |
| **Luiz** | Implementação e execução de testes integrados e validação de qualidade |
| **Natalia** | Análise exploratória de dados (EDA) e produção dos vídeos de apresentação |
| **Pedro** | Desenvolvimento de APIs (backend) e interface gráfica (frontend) |
| **Thiago** | Pesquisa, treinamento e validação dos modelos de Machine Learning |