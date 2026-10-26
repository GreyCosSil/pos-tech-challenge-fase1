# Modelo preditivo de churn para empresa do setor de Telecomunicaçoes

Breve descrição do que o modelo faz. Ex: "Modelo de classificação desenvolvido para prever a evasão de clientes (Churn) utilizando Random Forest, como trabalho de conclusão da pós-graduação."

## Estrutura do Repositório

* `src/`: Código fonte para pré-processamento, treinamento e predição.
* `notebooks/`: Análises exploratórias e protótipos de dados.
* `data/`: Diretório para armazenar os datasets localmente (ignorado pelo Git).
* `models/`: Diretório para os modelos treinados (ignorado pelo Git).
* `tests/`: Testes unitários do código.
* `docs/`: Documentação adicional.

## Como Configurar (Setup)

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/GreyCosSil/pos-tech-challenge-fase1.git
   cd pos-tech-challenge-fase1
   ```

## Verifique a versão do python instalado, Necessário o Python 3.11
   ```bash
   python3 --version
   ```

## Caso a versão não esteja correta necessario a instalacão
   ```bash
   brew reinstall python@3.11
   ```

## Recomenda-se criar ambiente virtual usando o python 3.11
   ```bash
   python3.11 -m venv .venv311
   source .venv311/bin/activate  # No Windows use: venv\Scripts\activate
   ```

<!-- # Atualize as ferramentas de instalação
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ``` -->
   
## Instala o projeto e as dependências de desenvolvimento
   ```bash
   python -m pip install -e '.[dev]'
```
## OU Instala as dependências com o requirements.txt
   ```bash
   pip install -r requirements.txt'
   ```
## Executar os testes automatizados com pytest
   ```bash
   pytest
   ```
## Subir o servidor da API com Uvicorn
   ```bash
   export JWT_SECRET="chave_secreta_local"
   uvicorn app.main:app --app-dir src --reload --host 0.0.0.0 --port 8000
   ```
## Execucao
### Carregar e visualizar os dados
   ```bash
   python src/main.py
   ```
### notebooks
   ```bash
   python src/eda.py
   ```
## Tests
   ```bash
   python -m pytest -q
   ```

<!-- ### Via Dockers
docker build -t meu-modelo-ml .
docker run -p 8000:8000 meu-modelo-ml -->
