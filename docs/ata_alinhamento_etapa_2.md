# Alinhamento da Reunião — Etapa 2

**Data:** 20 de outubro  
**Pauta:** Ajustes no modelo de Machine Learning

## 1. Métricas técnicas

Devem ser mantidas **apenas as métricas Precisão (Precision) e Recall**.

> **Observação:** O dataset é desbalanceado.

No contexto de churn em telecomunicações, acompanhar Precision e Recall é fundamental para avaliar o equilíbrio entre o custo de retenção e o custo da perda de clientes.

- **Recall (sensibilidade):** mede, entre todos os clientes que realmente cancelaram, quantos foram identificados pelo modelo. É importante para evitar que clientes em risco passem despercebidos.
- **Precision (precisão):** mede, entre todos os clientes apontados pelo modelo como churn, quantos realmente cancelaram. Ajuda a evitar gastos com ações de retenção direcionadas a clientes que não cancelariam.
- **Trade-off:** ao reduzir o *threshold* (limiar de decisão), o Recall tende a aumentar, pois mais clientes em risco são identificados. Em contrapartida, a Precision pode diminuir devido ao aumento de falsos positivos.

**Ação no projeto:** incluir `classification_report`, `precision_score` e `recall_score`, de `sklearn.metrics`, no ML Canvas e nos notebooks das Etapas 1 e 2.

## 2. Ajuste do threshold (limiar de decisão)

Por padrão, classificadores binários frequentemente usam um limiar de decisão de `0.5`: se a probabilidade estimada de churn for maior ou igual a 50%, o cliente será classificado como churn.

Em problemas de churn com dados desbalanceados, pode ser interessante avaliar limiares menores, como `0.30` ou `0.35`, para identificar mais clientes em risco. O valor ideal deve ser escolhido com base nos resultados de validação e nos custos de falsos positivos e falsos negativos.

**Onde aplicar:**
1. No notebook de experimentação (`notebooks/`).
2. Após a avaliação e seleção do limiar, na função de predição do código em `src/` que atende à API FastAPI (`/predict`).

Exemplo de aplicação de um limiar personalizado:

```python
# Obter as probabilidades da classe positiva (churn = 1)
y_probs = model.predict_proba(X_test)[:, 1]

# Aplicar o limiar escolhido (exemplo: 0.35)
custom_threshold = 0.35
y_pred_custom = (y_probs >= custom_threshold).astype(int)
```

## 3. Testar modelos com Random Forest e Boosting

A Etapa 2 exige testar e comparar diferentes modelos com o baseline de Regressão Logística.

**Modelos a avaliar:**
- Regressão Logística (baseline);
- `RandomForestClassifier` (solicitado no guia do projeto);
- Um algoritmo baseado em Boosting, como `XGBoost` ou `HistGradientBoostingClassifier`, do Scikit-Learn;
- `MLPClassifier`.

**Ação no projeto:** registrar os resultados em uma tabela comparativa entre Regressão Logística, Random Forest, Boosting e MLPClassifier, utilizando Precision e Recall como métricas principais. A comparação deve usar o mesmo conjunto de validação/teste e critérios consistentes.

## 4. Balanceamento das classes

A estratégia de balanceamento depende de onde será aplicada no pipeline.

### 4.1. Balanceamento dentro dos algoritmos

**Scikit-Learn (`RandomForestClassifier`, `LogisticRegression`):**

```python
model = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)
```

O parâmetro `class_weight="balanced"` ajusta automaticamente os pesos das classes de forma inversamente proporcional às suas frequências.

**XGBoost (`XGBClassifier`):**

```python
# Razão entre o total de exemplos negativos e positivos no conjunto de treino
ratio = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    scale_pos_weight=ratio,
    random_state=42
)
```

### 4.2. Balanceamento no pré-processamento (reamostragem / oversampling)

Caso a equipe opte por balancear os dados antes do treinamento, poderá utilizar bibliotecas como `imbalanced-learn`, com técnicas como `SMOTE` ou `RandomOverSampler`.

**Regra importante:** realizar a divisão dos dados primeiro (`train_test_split`) e aplicar a reamostragem **somente ao conjunto de treino** (`X_train`, `y_train`). Não reamostrar os conjuntos de validação ou teste, para evitar vazamento de dados (*data leakage*) e preservar uma avaliação realista.

## 5. Encaminhamentos

- [ ] Manter Precision e Recall como métricas principais.
- [ ] Incluir `classification_report`, `precision_score` e `recall_score` no ML Canvas e nos notebooks das Etapas 1 e 2.
- [ ] Testar diferentes thresholds no notebook de experimentação e selecionar o valor com base nos resultados de validação.
- [ ] Aplicar o threshold escolhido à função `/predict` da API FastAPI.
- [ ] Treinar e comparar Regressão Logística, Random Forest, Boosting e MLPClassifier.
- [ ] Avaliar estratégias de balanceamento por pesos de classe e/ou reamostragem apenas no conjunto de treino.
- [ ] Documentar os resultados em uma tabela comparativa.
