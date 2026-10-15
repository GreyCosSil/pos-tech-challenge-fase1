# Alinhamento da Reunião - Fase 1
**Data:** 13 de Outubro
**Pauta:** Ajustes no Canvas do Modelo de Machine Learning

---

## 🛠️ Ajustes Necessários no Canvas
* **Métricas Técnicas:** Devem ser mantidas **APENAS** as métricas de PR-AUC, Precisão e Recall. *(Observação importante: nosso dataset é desbalanceado).*
* **Data Collection:** A obtenção dos dados será feita via **API do Kaggle**.
* **Building Models:**
  * **Unificação:** Juntar as seções "Modelos tradicionais" e "Modelo principal" em uma única seção chamada apenas **Modelos**.
  * **Baselines:** Remover o *DummyClassifier*.
  * **Modelos:** Remover o *Gradient Boosting*. O framework principal será o **Scikit-learn**.
  * **Atualização do Modelo:** Estabelecer que o re-treinamento será **semanal** ou quando houver degradação de performance.

## 📌 Tarefas Individuais
* **Grey:** Verificar sobre o monitoramento do modelo e checar se o projeto exige a definição de Métricas de drift e Frequência de revisão.
* **Thiago:** Ajustar a base que vai ser usada no treinamento do modelo.
* **Pedro:** Ajustar a etapa de *Evaluate* considerando o *Precision* também no nosso modelo.

## ❓ Pontos de Atenção e Validação (Para decidir/verificar)
* **Impact Simulation:** Verificar se este tópico é realmente pedido. Se não for, remover.
* **Building Models:** Vamos utilizar árvores de decisão e redes neurais (MLPClassifier)?
* **Value Proposition:** O foco será voltado para *Customer Success*?

## ✅ Itens OK (Não precisam de alteração)
* KPIs de negócio
* Monitoramento operacional
* Features (Transformações)
* Data Sources
* Decisions
* Prediction Task