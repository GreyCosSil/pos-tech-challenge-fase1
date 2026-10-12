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

## Recomenda-se criar ambiente virtual
python3 -m venv venv
source venv/bin/activate  # No Windows use: venv\Scripts\activate

## instalacao de dependencias
pip install -r requirements.txt

## Execucao
<!-- ### Treino Modelo
python src/train.py -->

## Tests
python tests/test_main.py

<!-- ### Via Docker
docker build -t meu-modelo-ml .
docker run -p 8000:8000 meu-modelo-ml -->