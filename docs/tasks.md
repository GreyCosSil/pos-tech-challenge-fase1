# Tech Challenge — Fase 1
## Lista detalhada de tarefas

> Baseado no documento **“12MLET - Tech Challenge Fase 1.pdf”**.
>
> **Tema central:** Pipeline Preditivo de Churn com Modelagem Clássica e API REST.  
> **Entrega obrigatória:** Repositório GitHub + vídeo de até 5 minutos utilizando o método STAR.  
> **Entrega opcional:** Deploy em ambiente de produção em nuvem (AWS, Azure ou GCP).

---

## 1. Planejamento inicial e organização do projeto

- [x] Ler integralmente os requisitos do Tech Challenge.
- [x] Definir os integrantes e responsabilidades, caso o projeto seja realizado em grupo.
- [x] Definir o dataset que será utilizado. Preferencialmente, utilizar **Telco Customer Churn — IBM/Kaggle**.
- [x] Criar o repositório GitHub.
- [x] Criar a estrutura mínima exigida:
  - [x] `src/`
  - [x] `data/`
  - [x] `models/`
  - [x] `tests/`
  - [x] `notebooks/`
  - [x] `docs/`
- [x] Criar `.gitignore` apropriado para Python e Machine Learning.
- [x] Criar `README.md` inicial.
- [x] Configurar o gerenciamento de dependências (`requirements.txt`).
- [x] Definir a versão do Python.
- [x] Criar e documentar o ambiente virtual.
- [x] Instalar as bibliotecas mínimas necessárias ao projeto:
  - [x] Scikit-Learn
  - [x] Pandas
  - [x] NumPy
  - [x] Matplotlib
  - [x] FastAPI
  - [x] Uvicorn
  - [x] Pytest
  - [x] Joblib
- [x] Definir seed global para reprodutibilidade.
- [x] Estabelecer um padrão de commits claros e significativos.

**Critério de conclusão:** outra pessoa deve conseguir clonar o repositório, instalar as dependências e compreender a estrutura geral do projeto.

---

## 2. Definição do problema de negócio e ML Canvas

- [ ] Descrever claramente o problema de negócio.
- [ ] Definir o que significa `churn` no dataset escolhido.
- [ ] Identificar os stakeholders.
- [ ] Definir a decisão de negócio suportada pelo modelo.
- [ ] Definir a unidade de predição.
- [ ] Definir a variável-alvo.
- [ ] Identificar as principais features candidatas.
- [ ] Preencher o **ML Canvas**.
- [ ] Registrar no ML Canvas:
  - [ ] Problema de negócio
  - [ ] Usuários
  - [ ] Stakeholders
  - [ ] Fontes de dados
  - [ ] Features
  - [ ] Target
  - [ ] Riscos
  - [ ] Limitações
  - [ ] Métricas de negócio
  - [ ] Métricas de Machine Learning
- [ ] Definir uma métrica de negócio.
- [ ] Formalizar o objetivo técnico do modelo:
  - [ ] Estimar a probabilidade de churn.
  - [ ] Classificar clientes entre risco e não risco.

**Critério de conclusão:** deve existir uma formulação inequívoca de `X → y` e uma ligação clara entre a métrica técnica e o objetivo de negócio.

---

## 3. Análise Exploratória de Dados — EDA

- [ ] Carregar o dataset em notebook.
- [ ] Inspecionar:
  - [ ] Número de linhas
  - [ ] Número de colunas
  - [ ] Nomes das variáveis
  - [ ] Tipos das variáveis
  - [ ] Variável-alvo
- [ ] Criar um dicionário das variáveis.
- [ ] Verificar valores ausentes.
- [ ] Verificar duplicidades.
- [ ] Investigar valores inconsistentes.
- [ ] Identificar variáveis numéricas.
- [ ] Identificar variáveis categóricas.
- [ ] Identificar identificadores que não devem ser usados como features.
- [ ] Analisar a distribuição da variável `Churn`.
- [ ] Quantificar eventual desbalanceamento das classes.
- [ ] Gerar estatísticas descritivas das variáveis numéricas.
- [ ] Avaliar a cardinalidade das variáveis categóricas.
- [ ] Avaliar a relação das principais features com churn.
- [ ] Criar visualizações apropriadas.
- [ ] Investigar possíveis outliers.
- [ ] Investigar correlações entre variáveis numéricas.
- [ ] Verificar possíveis casos de **data leakage**.
- [ ] Documentar todas as decisões de limpeza.
- [ ] Consolidar a análise em um notebook, por exemplo:
  - [ ] `notebooks/01_eda.ipynb`
- [ ] Escrever conclusões da EDA, além de apresentar tabelas e gráficos.

**Critério de conclusão:** o notebook deve permitir compreender a qualidade dos dados, a distribuição do target e as principais características relacionadas ao churn.

---

## 4. Preparação e pré-processamento dos dados

- [ ] Separar features e target.
- [ ] Definir a estratégia de divisão entre treino e teste.
- [ ] Opcionalmente definir treino, validação e teste.
- [ ] Garantir divisão estratificada quando apropriado.
- [ ] Fixar `random_state`.
- [ ] Definir tratamento de valores ausentes.
- [ ] Definir transformação das features numéricas.
- [ ] Definir transformação das features categóricas.
- [ ] Avaliar a utilização de:
  - [ ] `SimpleImputer`
  - [ ] `StandardScaler`
  - [ ] `OneHotEncoder`
- [ ] Utilizar `ColumnTransformer` para centralizar as transformações, quando aplicável.
- [ ] Integrar pré-processamento e modelo por meio de `Pipeline` do Scikit-Learn.
- [ ] Garantir que os transformadores sejam ajustados somente com os dados de treino.
- [ ] Evitar vazamento de informações entre treino e teste.

**Critério de conclusão:** deve existir um pipeline de transformação reproduzível e sem vazamento de dados.

---

## 5. Definição das métricas de Machine Learning

- [ ] Escolher uma métrica técnica principal.
- [ ] Avaliar pelo menos:
  - [ ] Precision
  - [ ] Recall
  - [ ] F1-score
  - [ ] ROC-AUC
  - [ ] Matriz de confusão
- [ ] Justificar a métrica principal escolhida.
- [ ] Relacionar a métrica técnica ao problema de churn.
- [ ] Avaliar o custo de falso positivo.
- [ ] Avaliar o custo de falso negativo.
- [ ] Definir antecipadamente o critério de escolha do modelo campeão.
- [ ] Evitar utilizar somente acurácia caso exista desbalanceamento de classes.

**Critério de conclusão:** a equipe deve conseguir explicar por que a métrica escolhida é relevante para o problema de retenção de clientes.

---

## 6. Construção do baseline — Regressão Logística

- [ ] Construir o primeiro pipeline completo de Machine Learning.
- [ ] Utilizar `LogisticRegression`.
- [ ] Fixar seed quando aplicável.
- [ ] Treinar somente com o conjunto de treino.
- [ ] Gerar predições.
- [ ] Gerar probabilidades com `predict_proba`.
- [ ] Avaliar o modelo com as métricas definidas.
- [ ] Gerar matriz de confusão.
- [ ] Registrar os resultados.
- [ ] Documentar os hiperparâmetros utilizados.
- [ ] Documentar as limitações do baseline.
- [ ] Registrar as conclusões no notebook.

**Entregável da Etapa 1:** Notebook de EDA + baseline criado e documentado.

---

## 7. Treinamento de modelo baseado em árvores

- [ ] Selecionar um modelo baseado em árvore.
- [ ] Preferencialmente começar com `RandomForestClassifier`.
- [ ] Opcionalmente avaliar:
  - [ ] `DecisionTreeClassifier`
  - [ ] `GradientBoostingClassifier`
  - [ ] `HistGradientBoostingClassifier`
- [ ] Construir o treinamento de forma compatível com o pipeline de pré-processamento.
- [ ] Definir hiperparâmetros iniciais.
- [ ] Aplicar validação cruzada.
- [ ] Registrar:
  - [ ] Média das métricas
  - [ ] Desvio padrão das métricas
- [ ] Testar hiperparâmetros relevantes.
- [ ] Evitar tuning excessivo sem justificativa.
- [ ] Avaliar o modelo de forma comparável ao baseline.

**Critério de conclusão:** deve existir um resultado diretamente comparável ao modelo de Regressão Logística.

---

## 8. Treinamento da rede neural com MLPClassifier

- [ ] Criar pipeline com `MLPClassifier`.
- [ ] Garantir escalonamento adequado das features.
- [ ] Definir uma arquitetura inicial.
- [ ] Configurar os principais hiperparâmetros:
  - [ ] `hidden_layer_sizes`
  - [ ] `activation`
  - [ ] `solver`
  - [ ] `alpha`
  - [ ] `learning_rate`
  - [ ] `max_iter`
  - [ ] `random_state`
- [ ] Monitorar warnings de convergência.
- [ ] Ajustar `max_iter`, se necessário.
- [ ] Aplicar validação cruzada.
- [ ] Avaliar com as mesmas métricas utilizadas nos demais modelos.
- [ ] Registrar os resultados.

**Critério de conclusão:** devem estar disponíveis resultados comparáveis para três famílias de modelos: linear, árvore/ensemble e MLP.

---

## 9. Validação cruzada e comparação dos modelos

- [ ] Definir estratégia de cross-validation.
- [ ] Utilizar, por exemplo, `StratifiedKFold`.
- [ ] Fixar seed.
- [ ] Utilizar a mesma estratégia de validação para todos os modelos.
- [ ] Calcular as métricas de cada modelo.
- [ ] Registrar média e dispersão dos resultados.
- [ ] Consolidar os resultados em uma tabela comparativa.
- [ ] Comparar:
  - [ ] Logistic Regression
  - [ ] Modelo baseado em árvore/ensemble
  - [ ] MLPClassifier
- [ ] Considerar:
  - [ ] Desempenho preditivo
  - [ ] Variabilidade
  - [ ] Custo computacional
  - [ ] Interpretabilidade
  - [ ] Requisitos operacionais
- [ ] Selecionar o **modelo campeão**.
- [ ] Documentar a justificativa da escolha.

**Critério de conclusão:** deve existir uma tabela objetiva justificando a escolha do modelo campeão.

---

## 10. Treinamento final e persistência do modelo

- [ ] Definir os hiperparâmetros finais.
- [ ] Treinar novamente o pipeline campeão.
- [ ] Realizar a avaliação final.
- [ ] Registrar as métricas finais.
- [ ] Salvar o pipeline/modelo final.
- [ ] Utilizar um dos formatos permitidos:
  - [ ] `.joblib`
  - [ ] `.pkl`
- [ ] Armazenar o artefato em `models/`.
- [ ] Adotar uma nomenclatura clara, por exemplo:
  - [ ] `models/churn_pipeline.joblib`
- [ ] Registrar versão ou metadados do modelo.
- [ ] Verificar se o artefato pode ser carregado em um novo processo Python.

**Entregável da Etapa 2:** tabela comparativa de modelos + modelo final escolhido e salvo.

---

## 11. Refatoração do código produtivo para `src/`

- [ ] Remover dos notebooks toda lógica necessária à produção.
- [ ] Manter os notebooks para experimentação e análise.
- [ ] Criar módulos `.py` em `src/`.
- [ ] Separar responsabilidades, por exemplo:
  - [ ] Carregamento de dados
  - [ ] Pré-processamento
  - [ ] Treinamento
  - [ ] Avaliação
  - [ ] Persistência
  - [ ] Inferência
- [ ] Criar funções pequenas e coesas.
- [ ] Evitar duplicação de código.
- [ ] Utilizar type hints quando apropriado.
- [ ] Criar docstrings.
- [ ] Organizar constantes e configurações.
- [ ] Evitar caminhos absolutos.
- [ ] Garantir que o código produtivo funcione sem depender da execução prévia de notebooks.

**Critério de conclusão:** API e testes devem ser independentes da execução dos notebooks.

---

## 12. Implementação dos schemas da API

- [ ] Identificar as features necessárias na inferência.
- [ ] Criar modelos de entrada utilizando Pydantic.
- [ ] Definir os tipos das variáveis.
- [ ] Definir validações quando aplicável.
- [ ] Definir o schema de saída.
- [ ] Planejar uma resposta contendo:
  - [ ] Classificação de churn
  - [ ] Propensão/probabilidade de churn
- [ ] Definir o comportamento para entradas inválidas.

**Critério de conclusão:** o contrato de entrada e saída da API deve estar explicitamente definido.

---

## 13. Implementação da API com FastAPI

- [ ] Criar a aplicação FastAPI.
- [ ] Carregar o modelo persistido durante a inicialização da aplicação.
- [ ] Implementar `GET /health`.
- [ ] Retornar um status indicando que a API está operacional.
- [ ] Implementar `POST /predict`.
- [ ] Receber os dados do cliente.
- [ ] Converter os dados para o formato esperado pelo pipeline.
- [ ] Executar `predict`.
- [ ] Executar `predict_proba`, quando aplicável.
- [ ] Retornar a propensão de churn.
- [ ] Tratar erros adequadamente.
- [ ] Executar a aplicação localmente com Uvicorn.
- [ ] Testar a documentação automática:
  - [ ] `/docs`
  - [ ] `/redoc`

**Critério de conclusão:** uma requisição HTTP válida deve produzir uma previsão sem execução manual de notebooks.

---

## 14. Testes automatizados com Pytest

- [ ] Configurar `pytest`.
- [ ] Criar testes dentro de `tests/`.
- [ ] Criar pelo menos um teste da lógica de pré-processamento.
- [ ] Criar teste do endpoint `/health`.
- [ ] Idealmente criar teste do endpoint `/predict`.
- [ ] Testar código HTTP de resposta.
- [ ] Testar formato da resposta.
- [ ] Testar entrada válida.
- [ ] Testar pelo menos uma entrada inválida.
- [ ] Garantir que o modelo seja carregável.
- [ ] Executar:
  - [ ] `pytest`
- [ ] Garantir que todos os testes passem em ambiente limpo.

**Requisito mínimo do desafio:** pelo menos dois testes automatizados simples.

---

## 15. Garantia de reprodutibilidade

- [ ] Verificar que todas as seeds relevantes estejam fixadas.
- [ ] Garantir que `train_test_split` utilize `random_state`.
- [ ] Configurar seeds nos modelos relevantes.
- [ ] Verificar versões das dependências.
- [ ] Garantir que `pyproject.toml` ou `requirements.txt` esteja atualizado.
- [ ] Criar uma instalação limpa para teste.
- [ ] Reexecutar treinamento ou inferência.
- [ ] Confirmar resultados compatíveis.

---

## 16. Criação do Model Card

- [ ] Criar documento em `docs/`, por exemplo:
  - [ ] `docs/model_card.md`
- [ ] Descrever o objetivo do modelo.
- [ ] Informar o dataset utilizado.
- [ ] Informar a variável-alvo.
- [ ] Informar as features utilizadas.
- [ ] Informar o algoritmo campeão.
- [ ] Informar o processo de avaliação.
- [ ] Registrar as métricas.
- [ ] Registrar limitações.
- [ ] Registrar possíveis vieses.
- [ ] Explicar usos apropriados.
- [ ] Explicar usos inadequados.
- [ ] Registrar hipóteses e riscos.

**Critério de conclusão:** uma pessoa externa ao desenvolvimento deve compreender em quais condições o modelo pode ou não ser utilizado.

---

## 17. Finalização do README

- [ ] Criar descrição do problema.
- [ ] Explicar o objetivo do projeto.
- [ ] Informar o dataset.
- [ ] Mostrar a estrutura do repositório.
- [ ] Informar requisitos.
- [ ] Explicar criação do ambiente.
- [ ] Explicar instalação das dependências.
- [ ] Explicar execução dos notebooks.
- [ ] Explicar treinamento, se aplicável.
- [ ] Explicar execução da API.
- [ ] Mostrar exemplo de `/health`.
- [ ] Mostrar exemplo de `/predict`.
- [ ] Explicar execução dos testes.
- [ ] Apresentar resumo dos modelos testados.
- [ ] Apresentar o modelo campeão.
- [ ] Informar as métricas finais.
- [ ] Referenciar o Model Card.
- [ ] Identificar os integrantes do projeto.

---

## 18. Revisão da qualidade de engenharia

- [ ] Verificar a organização das pastas.
- [ ] Verificar a separação de responsabilidades.
- [ ] Remover código morto.
- [ ] Remover imports não utilizados.
- [ ] Remover prints de depuração.
- [ ] Verificar nomes de funções e variáveis.
- [ ] Verificar documentação.
- [ ] Certificar que nenhum segredo ou credencial esteja versionado.
- [ ] Certificar que datasets ou modelos muito grandes não estejam indevidamente versionados.
- [ ] Revisar `.gitignore`.
- [ ] Verificar a qualidade do histórico de commits.

> **Peso na avaliação:** Qualidade do Código e Engenharia — 25%.

---

## 19. Validação final da modelagem

- [ ] Confirmar que a EDA está completa.
- [ ] Confirmar a existência do baseline de Regressão Logística.
- [ ] Confirmar a existência do modelo baseado em árvore/ensemble.
- [ ] Confirmar a existência do `MLPClassifier`.
- [ ] Confirmar o uso de validação cruzada.
- [ ] Confirmar a comparação coerente entre os modelos.
- [ ] Confirmar a justificativa do modelo campeão.
- [ ] Confirmar as métricas finais.
- [ ] Conferir se o artefato final corresponde ao modelo campeão.

> **Peso na avaliação:** Modelagem de Machine Learning — 25%.

---

## 20. Validação final da API

- [ ] Subir a aplicação localmente.
- [ ] Executar `GET /health`.
- [ ] Executar `POST /predict` com entrada válida.
- [ ] Verificar retorno da probabilidade de churn.
- [ ] Testar múltiplos exemplos.
- [ ] Testar uma requisição inválida.
- [ ] Validar a documentação Swagger.
- [ ] Confirmar que a inicialização da API carrega o modelo corretamente.

> **Peso na avaliação:** API de Inferência — 20%.

---

## 21. Preparação do vídeo STAR

- [ ] Elaborar o roteiro antes da gravação.
- [ ] Manter a duração total em **até 5 minutos**.
- [ ] Estruturar o roteiro em quatro partes:
  - [ ] **Situation:** apresentar o problema de negócio e o contexto do dataset.
  - [ ] **Task:** explicar a tarefa do grupo e os objetivos técnicos.
  - [ ] **Action:** explicar limpeza de dados, escolha do modelo final e construção da API.
  - [ ] **Result:** apresentar os resultados obtidos com o modelo e demonstrar a API funcionando.
- [ ] Escolher as telas que serão demonstradas.
- [ ] Evitar mostrar código linha por linha.
- [ ] Mostrar rapidamente o repositório GitHub.
- [ ] Mostrar a tabela comparativa dos modelos.
- [ ] Mostrar o modelo escolhido.
- [ ] Mostrar `/docs` da FastAPI.
- [ ] Executar uma chamada `/predict`.
- [ ] Mostrar o retorno da API.
- [ ] Ensaiar para garantir duração menor ou igual a 5 minutos.
- [ ] Gravar a versão final.

> **Peso na avaliação:** Documentação e Vídeo STAR — 20%.

---

## 22. Deploy em nuvem — opcional

- [ ] Decidir se o deploy será realizado.
- [ ] Selecionar AWS, Azure ou GCP.
- [ ] Preparar a aplicação para produção.
- [ ] Opcionalmente criar `Dockerfile`.
- [ ] Configurar variáveis de ambiente.
- [ ] Publicar a API.
- [ ] Testar `/health` remotamente.
- [ ] Testar `/predict` remotamente.
- [ ] Documentar a URL e as instruções no README.

> O deploy em nuvem é opcional e deve ser priorizado somente após a conclusão dos requisitos obrigatórios.

---

## 23. Checklist final antes da entrega

- [ ] Repositório GitHub acessível.
- [ ] Estrutura `src/`, `data/`, `models/`, `tests/`, `notebooks/` e `docs/`.
- [ ] README completo.
- [ ] `.gitignore` correto.
- [ ] `pyproject.toml` ou `requirements.txt`.
- [ ] Dataset definido.
- [ ] EDA concluída.
- [ ] ML Canvas concluído.
- [ ] Métrica técnica definida.
- [ ] Métrica de negócio definida.
- [ ] Regressão Logística treinada.
- [ ] Modelo baseado em árvores treinado.
- [ ] MLPClassifier treinado.
- [ ] Validação cruzada realizada.
- [ ] Tabela comparativa disponível.
- [ ] Modelo campeão escolhido e justificado.
- [ ] Modelo `.pkl` ou `.joblib` persistido.
- [ ] Código produtivo em `src/`.
- [ ] Endpoint `/health` funcionando.
- [ ] Endpoint `/predict` funcionando.
- [ ] Pelo menos dois testes Pytest passando.
- [ ] Seeds definidas.
- [ ] Model Card concluído.
- [ ] Limitações e vieses documentados.
- [ ] Vídeo STAR gravado com duração ≤ 5 minutos.
- [ ] Clone limpo do repositório testado.
- [ ] Instalação das dependências testada.
- [ ] API executada a partir das instruções do README.
- [ ] `pytest` executado pela última vez.
- [ ] Links da entrega conferidos.

---

## Critérios de avaliação

| Critério | Peso |
|---|---:|
| Qualidade do Código e Engenharia | 25% |
| Modelagem de Machine Learning | 25% |
| API de Inferência | 20% |
| Testes e Reprodutibilidade | 10% |
| Documentação e Vídeo STAR | 20% |

---

## Sequência recomendada de execução

1. Setup do repositório e ambiente
2. ML Canvas
3. EDA
4. Pré-processamento
5. Baseline com Regressão Logística
6. Modelo baseado em árvores
7. MLPClassifier
8. Validação cruzada e comparação
9. Seleção e persistência do modelo campeão
10. Refatoração para `src/`
11. Implementação da FastAPI
12. Testes com Pytest
13. Model Card
14. README
15. Vídeo STAR
16. Revisão final
17. Deploy opcional
