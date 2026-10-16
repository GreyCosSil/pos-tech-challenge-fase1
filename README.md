# 🤖 Modelo preditivo de churn para empresa do setor de Telecomunicaçoes

Breve descrição do que o modelo faz. Ex: "Modelo de classificação desenvolvido para prever a evasão de clientes (Churn) utilizando Random Forest, como trabalho de conclusão da pós-graduação."

## 📁 Estrutura do Repositório

* `src/`: Código fonte para pré-processamento, treinamento e predição.
* `notebooks/`: Análises exploratórias e protótipos de dados.
* `data/`: Diretório para armazenar os datasets localmente (ignorado pelo Git).
* `models/`: Diretório para os modelos treinados (ignorado pelo Git).
* `tests/`: Testes unitários do código.
* `docs/`: Documentação adicional.

## 🚀 Como Configurar (Setup)

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
   
## instalacao de dependencias
   ```bash
   pip install -r requirements.txt
   ```

## Execucao
### Treino Modelo
   ```bash
   python src/train.py
   ```
### notebooks
   ```bash
   python notebooks/eda_natalia.py
   ```
## Tests
   ```bash
   python tests/test_main.py
   ```

<!-- ### Via Dockers
docker build -t meu-modelo-ml .
docker run -p 8000:8000 meu-modelo-ml -->