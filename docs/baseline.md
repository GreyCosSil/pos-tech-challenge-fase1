# Documentação técnica — Modelo baseline de Regressão Logística

## 1. Visão geral

**Objetivo:** estabelecer um modelo de referência (*baseline*) de classificação binária para prever a probabilidade de cancelamento de clientes (*customer churn*) de uma empresa fictícia de telecomunicações.

**Modelo:** sklearn.linear_model.LogisticRegression, integrado a um Pipeline de pré-processamento e estimativa. **Classe positiva:** cancelamento (Churn = Yes, codificada como 1). A seleção de hiperparâmetros maximiza o **Recall da classe positiva** na validação cruzada.

## 2. Base de dados

- **Dataset:** IBM Telco Customer Churn, disponível em [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
- **Dimensão:** 7.043 clientes e 21 colunas no CSV original.
- **Variável alvo:** Churn, codificada como No → 0 e Yes → 1.
- **Variáveis preditoras:** 19 colunas, excluindo customerID (identificador) e Churn (alvo).
- **Composição de classes:** 5.174 (No, 73,46%) e 1.869 (Yes, 26,54%). Há desbalanceamento de classes.

### 2.1. Verificação de integridade

| Verificação | Resultado observado |
|---|---:|
| Linhas inteiramente duplicadas | 0 |
| Identificadores customerID duplicados | 0 |
| TotalCharges ausentes após conversão | 11 (0,16%) |
| Valores não vazios inválidos em TotalCharges | 0 |

TotalCharges é convertido para numérico com pd.to_numeric(..., errors="coerce"). A categoria SeniorCitizen, embora representada numericamente na origem, é convertida para *string* e tratada como categórica. No internet service e No phone service são categorias válidas, não valores ausentes.

### 2.2. Variáveis utilizadas

**Numéricas:** tenure, MonthlyCharges, TotalCharges.

**Categóricas:** gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod.

## 3. Estratégia de particionamento

O conjunto de dados é dividido com train_test_split e os parâmetros:

| Parâmetro | Valor |
|---|---|
| test_size | 0.20 |
| random_state | 42 |
| Registros de treinamento | 5.634 |
| Registros de teste reservado | 1.409 |
| Percentual positivo no treinamento | 26,54% |

A estratificação preserva aproximadamente a proporção de cancelamentos em cada partição. O teste reservado não participa da seleção de hiperparâmetros.

## 4. Pré-processamento

O ColumnTransformer aplica tratamentos diferentes conforme o tipo de atributo.

| Grupo | Operações | Implementação |
|---|---|---|
| Numéricas | Imputação pela mediana; padronização | SimpleImputer(strategy="median") → StandardScaler() |
| Categóricas | Imputação pela moda; codificação one-hot | SimpleImputer(strategy="most_frequent") → OneHotEncoder(handle_unknown="ignore") |

## 5. Modelo estatístico

A Regressão Logística estima a probabilidade de pertencimento à classe positiva usando a função sigmoide.

O vetor x corresponde às variáveis transformadas pelo pré-processador. O estimador utiliza LogisticRegression(max_iter=5000, random_state=42).

## 6. Otimização e validação cruzada

O GridSearchCV avalia uma grade de **16 configurações** (4 valores de C × 4 configurações de class_weight), com validação cruzada StratifiedKFold(n_splits=5, shuffle=True, random_state=42).

| Hiperparâmetro | Valores testados |
|---|---|
| model__C | 0.01, 0.1, 1.0, 10.0 |
| model__class_weight | None, balanced, {0: 1, 1: 1.5}, {0: 1, 1: 2.0} |

**Parâmetros da busca:** scoring contém recall, average_precision, precision, f1, roc_auc, accuracy e balanced_accuracy; refit="recall"; n_jobs=1; return_train_score=False; error_score="raise".

A seleção pela maior média de Recall favorece a identificação de clientes que efetivamente cancelarão, ainda que isso gere mais alertas falsos positivos. class_weight="balanced" ajusta automaticamente a ponderação das classes de acordo com suas frequências no conjunto usado em cada ajuste.

### 6.1. Configuração selecionada

python
{
    "model__C": 1.0,
    "model__class_weight": "balanced",
}

## 7. Resultados experimentais

### 7.1. Validação cruzada

| Métrica | Média aproximada | Desvio-padrão aproximado |
|---|---:|---:|
| Recall | 0,80 | 0,04 |
| Average Precision (AP) | 0,66 | 0,02 |
| Precision | 0,52 | 0,02 |
| F1-score | 0,63 | 0,02 |
| ROC-AUC | 0,85 | 0,01 |
| Accuracy | 0,75 | 0,02 |
| Balanced Accuracy | 0,77 | 0,02 |

O modelo obtém o **Recall médio = 0,8013** para a melhor configuração. A tabela mostra as demais métricas.

O valor identificado como PR-AUC na descrição da análise corresponde operacionalmente ao scorer average_precision do scikit-learn, e não à integração trapezoidal da curva precisão–recall.

### 7.2. Teste reservado

| Métrica | Valor aproximado |
|---|---:|
| Recall da classe positiva | **0,78** |
| Average Precision (AP) | 0,63 |
| Precision da classe positiva | 0,50 |
| F1-score da classe positiva | 0,61 |
| ROC-AUC | 0,84 |
| Accuracy | 0,74 |
| Balanced Accuracy | 0,75 |

**Suporte por classe:** No: 1.035 clientes; Yes: 374 clientes. O relatório de classificação salvo apresenta:

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No (0) | 0,90 | 0,72 | 0,80 | 1.035 |
| Yes (1) | 0,50 | 0,78 | 0,61 | 374 |

A matriz de confusão é visualizada via ConfusionMatrixDisplay.from_predictions com os rótulos No e Yes, mas os valores absolutos de cada célula não são impressos textualmente nas saídas fornecidas; por esse motivo, não são reproduzidos como contagens exatas nesta documentação.

### 7.3. Interpretação

- O Recall de cerca de 78% indica que o modelo detecta a maior parte dos cancelamentos reais no teste.
- A Precision de cerca de 50% indica que aproximadamente metade dos clientes sinalizados como potenciais cancelamentos realmente pertence à classe positiva.
- A ROC-AUC de aproximadamente 0,84 indica capacidade discriminativa entre as duas classes.
- A Balanced Accuracy de aproximadamente 0,75 é mais informativa que Accuracy isolada quando há desbalanceamento.
- A proximidade entre as métricas de validação cruzada e de teste é compatível com generalização razoável para esse particionamento, mas não constitui comprovação de robustez em outros períodos, populações ou ambientes operacionais.
