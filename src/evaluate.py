"""
Avaliação de modelos de classificação JÁ TREINADOS (Recall e PR-AUC).

O treinamento acontece em ``src/train.py``. Este módulo apenas carrega e
avalia os modelos, sem treinar nem alterar nenhum deles.

Uso típico:
    from src.avaliacao import avaliar_modelo, carregar_modelos

    modelos = carregar_modelos("models/")          # ou um dict {nome: modelo}
    resultados = avaliar_modelo(modelos, X_teste, y_teste)
    print(resultados)
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


# --------------------------------------------------------------------------- #
# Carregamento
# --------------------------------------------------------------------------- #
def carregar_modelos(diretorio="models", extensoes=(".joblib", ".pkl")):
    """Carrega todos os modelos salvos (com ``joblib.dump``) em um diretório.

    O nome de cada modelo é o nome do arquivo sem extensão,
    ex.: ``models/random_forest.joblib`` → ``"random_forest"``.

    Retorna um dicionário {nome: modelo}.
    """
    diretorio = Path(diretorio)
    arquivos = sorted(p for p in diretorio.iterdir() if p.suffix in extensoes)
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum modelo {extensoes} encontrado em '{diretorio}'. "
            "Rode 'python src/train.py' primeiro."
        )
    return {arquivo.stem: joblib.load(arquivo) for arquivo in arquivos}


# --------------------------------------------------------------------------- #
# Funções auxiliares
# --------------------------------------------------------------------------- #
def _normalizar_modelos(modelos):
    """Aceita dicionário {nome: modelo} ou lista de modelos."""
    if isinstance(modelos, dict):
        return dict(modelos)

    normalizados = {}
    for modelo in modelos:
        estimador = modelo.steps[-1][1] if isinstance(modelo, Pipeline) else modelo
        base = nome = type(estimador).__name__
        i = 2
        while nome in normalizados:  # evita colisão de nomes
            nome, i = f"{base}_{i}", i + 1
        normalizados[nome] = modelo
    return normalizados


def _validar_treinado(nome, modelo):
    """Garante que o modelo foi treinado antes de ser avaliado."""
    try:
        check_is_fitted(modelo)
    except NotFittedError as erro:
        raise NotFittedError(
            f"O modelo '{nome}' não está treinado. Treine-o em train.py "
            "antes de avaliá-lo."
        ) from erro


def _obter_scores(modelo, X):
    """Retorna o score contínuo da classe positiva (usado na PR-AUC)."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    if hasattr(modelo, "decision_function"):
        return modelo.decision_function(X)
    raise TypeError(
        f"{type(modelo).__name__} não possui predict_proba nem decision_function; "
        "não é possível calcular a PR-AUC."
    )


def _prever_classes(modelo, X, scores, limiar):
    """Converte scores em classes usando o limiar (quando há probabilidades)."""
    if hasattr(modelo, "predict_proba"):
        return (scores >= limiar).astype(int)
    return modelo.predict(X)  # decision_function: usa o corte padrão do modelo


def _calcular_metricas(y_true, y_pred, scores):
    return {
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "pr_auc": average_precision_score(y_true, scores, pos_label=1),
    }


# --------------------------------------------------------------------------- #
# Função principal
# --------------------------------------------------------------------------- #
def avaliar_modelo(modelos, X_teste, y_teste, limiar=0.5, ordenar_por="recall"):
    """Avalia classificadores binários já treinados com Recall e PR-AUC.

    Parâmetros
    ----------
    modelos : dict {nome: modelo} ou lista de modelos scikit-learn treinados
        (Regressão Logística, Árvore de Decisão, Random Forest, MLP, Pipelines…).
    X_teste, y_teste : dados de teste. A classe positiva (churn) deve ser 1.
    limiar : limiar de decisão sobre a probabilidade, usado no Recall.
    ordenar_por : coluna usada para ordenar o resultado ("pr_auc" ou "recall").

    Retorna
    -------
    pd.DataFrame com uma linha por modelo e as colunas ``recall``, ``pr_auc``
    e ``pr_auc_baseline`` (proporção de positivos no teste, que é a PR-AUC de
    um modelo aleatório: um modelo útil deve ficar bem acima dela).

    Exceções
    --------
    NotFittedError : se algum modelo não tiver sido treinado.
    """
    modelos = _normalizar_modelos(modelos)
    if not modelos:
        raise ValueError("Nenhum modelo foi informado.")
    if ordenar_por not in ("recall", "pr_auc"):
        raise ValueError("ordenar_por deve ser 'recall' ou 'pr_auc'.")

    y_teste = np.asarray(y_teste)
    baseline = float(np.mean(y_teste == 1))

    linhas = []
    for nome, modelo in modelos.items():
        _validar_treinado(nome, modelo)
        scores = _obter_scores(modelo, X_teste)
        y_pred = _prever_classes(modelo, X_teste, scores, limiar)
        linhas.append({"modelo": nome, **_calcular_metricas(y_teste, y_pred, scores)})

    return (
        pd.DataFrame(linhas)
        .set_index("modelo")
        .sort_values(ordenar_por, ascending=False)
        .assign(pr_auc_baseline=baseline)
        .round(4)
    )


# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #
# Paleta Okabe-Ito: distinguível por quem tem daltonismo.
_CORES = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def plotar_pr_auc(
    modelos,
    X_teste,
    y_teste,
    salvar_em=None,
    mostrar=True,
    titulo="Curva Precisão-Revocação (PR) por modelo",
    tabela=True,
    limiar=0.5,
):
    """Plota as curvas Precisão-Revocação de modelos já treinados e, ao final,
    a tabela comparativa dos modelos.

    Cada curva mostra a PR-AUC (average precision) na legenda. A linha
    tracejada é o baseline: a proporção de positivos no teste, que é o
    desempenho de um modelo aleatório.

    Parâmetros
    ----------
    modelos : dict {nome: modelo} ou lista de modelos treinados
        (os mesmos aceitos por ``avaliar_modelo``).
    X_teste, y_teste : dados de teste. A classe positiva (churn) deve ser 1.
    salvar_em : caminho para salvar a imagem das curvas (ex.: "docs/pr_auc.png").
        A tabela é salva ao lado, com o sufixo ``_tabela`` (ex.: "docs/recall_pr_auc_tabela.png").
    mostrar : se True, chama ``plt.show()`` (uma vez, após criar as figuras).
    titulo : título do gráfico das curvas.
    tabela : se True (padrão), plota a tabela comparativa ao final.
    limiar : limiar de decisão usado no Recall da tabela.

    Retorna
    -------
    matplotlib.figure.Figure com as curvas. A tabela, quando gerada, fica em
    uma segunda figura (use ``plotar_tabela_comparativa`` para obtê-la).
    """
    modelos = _normalizar_modelos(modelos)
    if not modelos:
        raise ValueError("Nenhum modelo foi informado.")

    y_teste = np.asarray(y_teste)
    baseline = float(np.mean(y_teste == 1))

    # Ordena pela PR-AUC para a legenda ficar do melhor para o pior modelo
    curvas = []
    for nome, modelo in modelos.items():
        _validar_treinado(nome, modelo)
        scores = _obter_scores(modelo, X_teste)
        precisao, revocacao, _ = precision_recall_curve(y_teste, scores, pos_label=1)
        pr_auc = average_precision_score(y_teste, scores, pos_label=1)
        curvas.append((nome, precisao, revocacao, pr_auc))
    curvas.sort(key=lambda c: c[3], reverse=True)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    for i, (nome, precisao, revocacao, pr_auc) in enumerate(curvas):
        ax.plot(
            revocacao,
            precisao,
            drawstyle="steps-post",
            linewidth=2,
            color=_CORES[i % len(_CORES)],
            label=f"{nome} (PR-AUC = {pr_auc:.3f})",
        )

    ax.axhline(
        baseline,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label=f"Baseline aleatório ({baseline:.3f})",
    )

    ax.set_xlabel("Revocação (Recall)")
    ax.set_ylabel("Precisão (Precision)")
    ax.set_title(titulo)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    fig.tight_layout()

    if salvar_em:
        fig.savefig(salvar_em, dpi=150, bbox_inches="tight")

    # Ao final, a tabela comparativa dos modelos
    if tabela:
        caminho_tabela = None
        if salvar_em:
            caminho = Path(salvar_em)
            caminho_tabela = caminho.with_name(f"{caminho.stem}_tabela{caminho.suffix}")
        plotar_tabela_comparativa(
            modelos,
            X_teste,
            y_teste,
            limiar=limiar,
            salvar_em=caminho_tabela,
            mostrar=False,
        )

    if mostrar:
        plt.show()
    return fig


def plotar_tabela_comparativa(
    modelos,
    X_teste,
    y_teste,
    limiar=0.5,
    salvar_em=None,
    mostrar=True,
    titulo="Comparativo dos modelos",
):
    """Plota uma tabela comparando os modelos (Recall e PR-AUC).

    O melhor valor de cada métrica aparece em negrito com fundo destacado, e
    os modelos ficam ordenados pelo RECALL. A coluna "Ganho vs baseline" diz
    quantas vezes a RECALL do modelo supera a de um classificador aleatório.

    Parâmetros
    ----------
    modelos : dict {nome: modelo} ou lista de modelos já treinados.
    X_teste, y_teste : dados de teste. A classe positiva (churn) deve ser 1.
    limiar : limiar de decisão usado no Recall.
    salvar_em : caminho para salvar a imagem. Opcional.
    mostrar : se True, chama ``plt.show()``.

    Retorna
    -------
    matplotlib.figure.Figure
    """
    resultados = avaliar_modelo(modelos, X_teste, y_teste, limiar=limiar)

    colunas = ["Modelo", "Recall", "PR-AUC", "Baseline", "Ganho vs baseline"]
    linhas = [
        [
            nome,
            f"{r.recall:.3f}",
            f"{r.pr_auc:.3f}",
            f"{r.pr_auc_baseline:.3f}",
            f"{r.pr_auc / r.pr_auc_baseline:.1f}x",
        ]
        for nome, r in resultados.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(8, 0.9 + 0.5 * len(linhas)))
    ax.axis("off")
    ax.set_title(f"{titulo} (limiar = {limiar})", fontsize=12, pad=10)

    tabela = ax.table(
        cellText=linhas,
        colLabels=colunas,
        cellLoc="center",
        loc="center",
        colWidths=[0.30, 0.14, 0.14, 0.14, 0.22],
    )
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10.5)
    tabela.scale(1, 1.7)

    # Cabeçalho
    for j in range(len(colunas)):
        celula = tabela[0, j]
        celula.set_facecolor("#2F3B4C")
        celula.set_text_props(color="white", fontweight="bold")

    # Destaca o melhor valor de Recall (coluna 1) e de PR-AUC (coluna 2)
    for j, coluna in ((1, "recall"), (2, "pr_auc")):
        melhor = resultados[coluna].values.argmax() + 1  # +1: pula o cabeçalho
        tabela[melhor, j].set_facecolor("#CDEBDD")
        tabela[melhor, j].set_text_props(fontweight="bold")

    # Alinha o nome do modelo à esquerda
    for i in range(1, len(linhas) + 1):
        tabela[i, 0].set_text_props(ha="left")
        tabela[i, 0]._loc = "left"

    fig.tight_layout()
    if salvar_em:
        fig.savefig(salvar_em, dpi=150, bbox_inches="tight")
    if mostrar:
        plt.show()
    return fig
