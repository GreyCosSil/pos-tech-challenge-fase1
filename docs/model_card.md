# Model Card — Predição de cancelamento de clientes de telecomunicações

## 1. Resumo executivo

| Campo | Descrição |
|---|---|
| Nome do caso de uso | Telco Customer Churn Prediction |
| Objetivo | Estimar a propensão de um cliente cancelar serviços de telecomunicações (`Churn`). |
| Tipo de tarefa | Classificação binária supervisionada, com estimativa de probabilidade da classe positiva. |
| Modelo selecionado no experimento comparativo | **Random Forest** (`RandomForest`), selecionado pelo maior *Recall* médio na validação cruzada. |
| Modelo de referência (*baseline*) | **Regressão Logística** (`LogisticRegression`). |
| Classe positiva | `Churn = Yes` → `1`; classe negativa: `Churn = No` → `0`. |
| Fonte dos dados | Conjunto **Telco Customer Churn**. |
| Dimensão da base | 7.043 registros, 21 colunas originais. |
| Dados de desenvolvimento | 5.634 clientes em treino (80%) e 1.409 clientes em teste reservado (20%), com estratificação. |
| Métrica de seleção | Maior *Recall* médio na validação cruzada. |
| Resultado de teste do Random Forest | **Recall = 0,7834**; **Precision = 0,4811**; **F1 = 0,5961**; **Accuracy = 0,7182** (valores calculados a partir da matriz de confusão). |

## 2. Finalidade

- **Pesquisa e comparação de algoritmos** de classificação tabular para churn.
- **Apoio analítico** à priorização de clientes para campanhas de retenção, condicionado à validação em dados reais e à avaliação dos custos operacionais.
- **Estudo de sensibilidade** entre *Recall*, *Precision*, *Average Precision* e volume de clientes sinalizados.

## 3. Dados utilizados

### 3.1. Origem e composição

O dataset corresponde a uma base educacional/de exemplo de clientes de telecomunicações.

| Indicador | Valor observado |
|---|---:|
| Total de clientes | 7.043 |
| Colunas originais | 21 |
| Preditores utilizados | 19 |
| `Churn = No` | 5.174 (73,46%) |
| `Churn = Yes` | 1.869 (26,54%) |
| Linhas duplicadas | 0 |
| Identificadores `customerID` duplicados | 0 |
| Valores ausentes em `TotalCharges` | 11 (0,16%) |
| Valores inválidos não vazios em `TotalCharges` | 0 |
| Demais valores ausentes identificados | Nenhum na inspeção registrada |

**Desbalanceamento:** a classe positiva representa aproximadamente 26,54% da base. Portanto, a acurácia isolada não é suficiente para avaliar a capacidade de detectar cancelamentos.

### 3.2. Atributos preditores

**Numéricos (3):** `tenure`, `MonthlyCharges`, `TotalCharges`.

**Categóricos (16):** `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`.

A coluna `customerID` é excluída por ser um identificador, e `Churn` é utilizada exclusivamente como alvo. `SeniorCitizen`, originalmente inteira (`0`/`1`), é tratada como categoria no baseline. As categorias `No internet service` e `No phone service` são valores legítimos, não dados faltantes.

### 3.3. Particionamento e validação

- Divisão treino/teste: **80%/20%**, com `stratify=y`.
- Treinamento: **5.634** registros; teste: **1.409** registros.
- Proporção de churn no treinamento: **26,54%**.
- Validação interna: **cinco folds estratificados**, conforme documentação dos notebooks.

## 4. Pré-processamento

O notebook do baseline implementa explicitamente o seguinte fluxo com `Pipeline` e `ColumnTransformer` do scikit-learn:

| Tipo | Tratamento |
|---|---|
| Atributos numéricos | Imputação pela mediana (`SimpleImputer(strategy="median")`) e padronização (`StandardScaler`). |
| Atributos categóricos | Imputação pela moda (`SimpleImputer(strategy="most_frequent")`) e codificação *one-hot* (`OneHotEncoder(handle_unknown="ignore")`). |
| `TotalCharges` | Conversão para número com `pd.to_numeric(..., errors="coerce")` antes do pipeline. |
| Alvo `Churn` | `No → 0` e `Yes → 1`. |

## 5. Arquitetura e treinamento

### 5.1. Algoritmos avaliados

1. Regressão Logística (`LogisticRegression`).
2. Árvore de Decisão (`DecisionTree`).
3. Random Forest (`RandomForest`).
4. Rede neural multicamadas (`MLPClassifier`).

### 5.2. Configuração selecionada — Random Forest

Parâmetros registrados:

```yaml
algorithm: RandomForest
class_weight: balanced
max_depth: 4
max_features: sqrt
min_samples_leaf: 16
n_estimators: 200
```

### 5.3. Configuração do baseline — Regressão Logística

`GridSearchCV` com:

- `C ∈ {0.01, 0.1, 1.0, 10.0}`.
- `class_weight ∈ {None, "balanced", {0: 1, 1: 1.5}, {0: 1, 1: 2.0}}`.
- `max_iter = 5000`, `random_state = 42`.
- Métrica de refit: `recall`.

**Melhores parâmetros registrados:** `C = 1.0` e `class_weight = "balanced"`; *Recall* médio na validação cruzada: **0,8013**.

## 6. Desempenho e avaliação

### 6.1. Validação cruzada — comparação dos quatro algoritmos

| Modelo | Recall ↑ | Average Precision ↑ | Precision ↑ | F1 ↑ | ROC-AUC ↑ | Accuracy ↑ | Balanced Accuracy ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Random Forest** | **0,81 ± 0,03** | 0,65 ± 0,01 | 0,50 ± 0,01 | 0,62 ± 0,01 | 0,84 ± 0,01 | 0,74 ± 0,01 | 0,76 ± 0,01 |
| Regressão Logística | 0,80 ± 0,04 | **0,66 ± 0,02** | 0,52 ± 0,02 | **0,63 ± 0,02** | **0,85 ± 0,01** | 0,75 ± 0,02 | **0,77 ± 0,02** |
| Árvore de Decisão | 0,79 ± 0,06 | 0,60 ± 0,02 | 0,49 ± 0,02 | 0,60 ± 0,01 | 0,83 ± 0,01 | 0,72 ± 0,02 | 0,75 ± 0,01 |
| MLP | 0,56 ± 0,06 | 0,65 ± 0,02 | **0,66 ± 0,03** | 0,60 ± 0,03 | 0,84 ± 0,01 | **0,80 ± 0,01** | 0,72 ± 0,02 |

### 6.2. Teste reservado — Random Forest versus baseline

| Métrica | Random Forest (selecionado) | Regressão Logística (baseline) |
|---|---:|---:|
| **Recall / sensibilidade** | **0,7834** | **0,7834** |
| Precision | 0,4811 | **0,5034** |
| F1-score | 0,5961 | **0,6130** |
| Average Precision (saída arredondada) | **0,64** | 0,63 |
| ROC-AUC (saída arredondada) | 0,84 | 0,84 |
| Accuracy | 0,7182 | **0,7374** |
| Balanced Accuracy | 0,7391 | **0,7521** |
| Especificidade (calculada) | 0,6947 | **0,7208** |

### 6.3. Matrizes de confusão — conjunto de teste

Linhas: classe real. Colunas: classe prevista. Classe positiva: `Churn = Yes`.

**Random Forest**

| Real \ Previsto | No | Yes |
|---|---:|---:|
| **No** | 719 (VN) | 316 (FP) |
| **Yes** | 81 (FN) | 293 (VP) |

**Regressão Logística**

| Real \ Previsto | No | Yes |
|---|---:|---:|
| **No** | 746 (VN) | 289 (FP) |
| **Yes** | 81 (FN) | 293 (VP) |

Suporte: **1.035 clientes `No` e 374 clientes `Yes`**, totalizando 1.409 no teste.

### 6.4. Perfil de erros e implicações operacionais

| Indicador de teste | Random Forest | Baseline |
|---|---:|---:|
| Cancelamentos identificados corretamente (VP) | 293 | 293 |
| Cancelamentos não detectados (FN) | 81 | 81 |
| Clientes sem cancelamento indevidamente sinalizados (FP) | 316 | 289 |
| Clientes sem cancelamento corretamente descartados (VN) | 719 | 746 |
| Total sinalizado para possível ação de retenção (VP + FP) | 609 | 582 |

No ponto de operação registrado, o Random Forest gera **27 sinalizações positivas adicionais e todas elas correspondem a falsos positivos**. A Regressão Logística apresenta, nesse conjunto de teste e nesse limiar de decisão, um resultado operacional mais favorável para identificar o mesmo número de cancelamentos com menos abordagens.

## 7. Conclusão e decisão recomendada

**Modelo vencedor segundo o critério original:** Random Forest, pela maior média de *Recall* na validação cruzada. **Melhor resultado de identificação no teste:** empate em *Recall* (293 verdadeiros positivos e 81 falsos negativos para ambos).

**Conclusão da avaliação comparativa:** a Regressão Logística apresenta menos falsos positivos e melhores *Precision*, F1, *Accuracy*, especificidade e *Balanced Accuracy* no limiar avaliado, enquanto o Random Forest apresenta *Average Precision* ligeiramente superior na saída arredondada do teste.

**Recomendação:** manter ambos como candidatos experimentais; usar a Regressão Logística como referência operacional inicial para análises de custo/benefício; somente definir um modelo de produção após revalidação, calibração, escolha de limiar, testes de equidade e avaliação de negócio.
