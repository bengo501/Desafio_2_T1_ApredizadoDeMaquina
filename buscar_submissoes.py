"""
busca de hiperparâmetros e geração de vários arquivos de submissão.

limite importante (enunciado + kaggle):
  a métrica de 87% é calculada no site da competição com rótulos ocultos.
  este script só maximiza um **proxy** (holdout ou validação cruzada) com
  **knn**, **multinomial naive bayes** e **árvore de decisão** + vetorização.

com só esses três classificadores e bag-of-words/tf-idf, o proxy local neste
conjunto costuma **estacionar perto de 0,62–0,64**; rodar mais tentativas
ajuda a **varrer** o espaço e pode melhorar um pouco, mas **não garante**
proxy0,87 nem score0,87 no kaggle — isso só o site informa após envio.

fluxo:
  1) rode com --trials alto e, se quiser, --modo cv;
  2) use manifest.csv (ordenado) e/ou --top-k;
  3) envie ao kaggle começando pelos melhores proxies.

exemplo (busca grande + pasta nova para não misturar runs):
  python buscar_submissoes.py --out-dir submissoes_candidatas02 --trials 900 --copiar-melhor-para submission.csv --top-k 10
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import warnings
from pathlib import Path
from typing import Iterator, Tuple

import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UserWarning)

TAG_RE = re.compile(r"<[^>]+>")


def limpar_texto(texto) -> str:
    if not isinstance(texto, str):
        texto = str(texto)
    texto = TAG_RE.sub(" ", texto)
    texto = texto.lower()
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def iter_grade_prioritaria() -> Iterator[Tuple[str, object, str, object]]:
    """
    percorre combinações inspiradas no notebook (onde count+mnb com min_df=2,
    max_df=0.95 e stop_words english costuma ir bem).
    """
    alphas_mnb = [
        0.0005,
        0.001,
        0.003,
        0.005,
        0.008,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.1,
        0.11,
        0.12,
        0.13,
        0.14,
        0.15,
        0.175,
        0.2,
        0.25,
        0.3,
        0.4,
        0.5,
        0.65,
        0.8,
        1.0,
        1.5,
        2.0,
    ]

    configs_count = [
        dict(
            ngram_range=(1, 1),
            min_df=2,
            max_df=0.95,
            stop_words="english",
            max_features=None,
        ),
        dict(
            ngram_range=(1, 1),
            min_df=2,
            max_df=0.95,
            stop_words="english",
            max_features=20_000,
        ),
        dict(
            ngram_range=(1, 1),
            min_df=2,
            max_df=0.92,
            stop_words="english",
            max_features=25_000,
        ),
        dict(
            ngram_range=(1, 1),
            min_df=3,
            max_df=0.95,
            stop_words="english",
            max_features=None,
        ),
        dict(
            ngram_range=(1, 2),
            min_df=2,
            max_df=1.0,
            stop_words="english",
            max_features=20_000,
        ),
        dict(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words="english",
            max_features=35_000,
        ),
        dict(
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.92,
            stop_words="english",
            max_features=40_000,
        ),
        dict(
            ngram_range=(1, 3),
            min_df=3,
            max_df=0.9,
            stop_words="english",
            max_features=30_000,
        ),
        dict(ngram_range=(1, 1), min_df=1, max_df=1.0, stop_words=None, max_features=None),
        dict(ngram_range=(1, 1), min_df=1, max_df=0.88, stop_words="english", max_features=50_000),
        dict(ngram_range=(1, 2), min_df=2, max_df=0.9, stop_words="english", max_features=30_000),
        dict(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words="english",
            max_features=20_000,
            binary=True,
        ),
    ]

    for cfg in configs_count:
        v = CountVectorizer(**cfg)
        for a in alphas_mnb:
            yield "count", v, "mnb", MultinomialNB(alpha=a)

    configs_tfidf = [
        dict(
            ngram_range=(1, 1),
            min_df=2,
            max_features=15_000,
            stop_words="english",
            sublinear_tf=True,
            max_df=1.0,
        ),
        dict(
            ngram_range=(1, 1),
            min_df=2,
            max_features=25_000,
            stop_words="english",
            sublinear_tf=True,
            max_df=0.95,
        ),
        dict(
            ngram_range=(1, 1),
            min_df=1,
            max_features=40_000,
            stop_words="english",
            sublinear_tf=False,
            max_df=0.9,
        ),
        dict(
            ngram_range=(1, 2),
            min_df=2,
            max_features=20_000,
            stop_words="english",
            sublinear_tf=True,
            max_df=0.95,
        ),
        dict(
            ngram_range=(1, 2),
            min_df=2,
            max_features=45_000,
            stop_words="english",
            sublinear_tf=True,
            max_df=0.92,
        ),
        dict(
            ngram_range=(1, 3),
            min_df=2,
            max_features=35_000,
            stop_words="english",
            sublinear_tf=True,
            max_df=0.9,
        ),
        dict(
            ngram_range=(1, 3),
            min_df=3,
            max_features=28_000,
            stop_words="english",
            sublinear_tf=True,
            max_df=0.88,
        ),
    ]
    for cfg in configs_tfidf:
        v = TfidfVectorizer(**cfg)
        for a in alphas_mnb:
            yield "tfidf", v, "mnb", MultinomialNB(alpha=a)

    # knn: vetor count razoável + vários k (métrica cosine costuma ser mais estável em texto esparsa)
    vet_knn = CountVectorizer(
        ngram_range=(1, 1),
        min_df=2,
        max_df=0.95,
        stop_words="english",
    )
    for k in (3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17, 21):
        for w in ("uniform", "distance"):
            yield "count", clone(vet_knn), "knn", KNeighborsClassifier(
                n_neighbors=k,
                weights=w,
                metric="cosine",
                algorithm="brute",
                n_jobs=-1,
            )

    vet_knn_big = CountVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words="english",
        max_features=18_000,
    )
    for k in (5, 7, 9, 11, 15):
        for w in ("uniform", "distance"):
            yield "count", clone(vet_knn_big), "knn", KNeighborsClassifier(
                n_neighbors=k,
                weights=w,
                metric="cosine",
                algorithm="brute",
                n_jobs=-1,
            )

    vet_knn2 = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=20_000,
        stop_words="english",
        sublinear_tf=True,
    )
    for k in (3, 5, 7, 9, 11, 15, 19):
        for w in ("uniform", "distance"):
            yield "tfidf", clone(vet_knn2), "knn", KNeighborsClassifier(
                n_neighbors=k,
                weights=w,
                metric="cosine",
                algorithm="brute",
                n_jobs=-1,
            )

    vet_knn3 = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=35_000,
        stop_words="english",
        sublinear_tf=True,
        max_df=0.92,
    )
    for k in (7, 11, 15):
        yield "tfidf", clone(vet_knn3), "knn", KNeighborsClassifier(
            n_neighbors=k,
            weights="distance",
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )

    vet_tree = CountVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=15_000,
        stop_words="english",
    )
    for depth in (8, 10, 12, 15, 18, 20, 25, 30, 40, None):
        for mss in (2, 5, 10):
            for msl in (1, 2, 4):
                for crit in ("gini", "entropy"):
                    yield "count", clone(vet_tree), "tree", DecisionTreeClassifier(
                        max_depth=depth,
                        min_samples_split=mss,
                        min_samples_leaf=msl,
                        criterion=crit,
                        random_state=RANDOM_STATE,
                    )

    vet_tree2 = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=12_000,
        stop_words="english",
        sublinear_tf=True,
    )
    for depth in (12, 18, 25, 35, None):
        for mss in (2, 5):
            for crit in ("gini", "entropy"):
                yield "tfidf", clone(vet_tree2), "tree", DecisionTreeClassifier(
                    max_depth=depth,
                    min_samples_split=mss,
                    min_samples_leaf=2,
                    criterion=crit,
                    random_state=RANDOM_STATE,
                )


def amostrar_vetorizador(rng: random.Random):
    kind = rng.choice(["count", "tfidf"])
    ngram = rng.choice([(1, 1), (1, 2), (1, 3)])
    min_df = rng.choice([1, 2, 3, 4, 5])
    max_df = rng.choice([0.8, 0.85, 0.88, 0.9, 0.92, 0.95, 1.0])
    stop = rng.choice([None, "english"])
    max_feat = rng.choice(
        [None, 8_000, 12_000, 15_000, 20_000, 30_000, 45_000, 60_000, 80_000]
    )

    common = dict(
        ngram_range=ngram,
        min_df=min_df,
        max_df=max_df,
        max_features=max_feat,
        stop_words=stop,
    )
    if kind == "count":
        if rng.random() < 0.12:
            common["binary"] = True
        return kind, CountVectorizer(**common)
    sublinear = rng.choice([True, False])
    norm = rng.choice([None, "l2"])
    return kind, TfidfVectorizer(sublinear_tf=sublinear, norm=norm, **common)


def amostrar_classificador(rng: random.Random):
    """peso maior em mnb (costuma liderar o proxy neste tipo de tarefa)."""
    nome = rng.choices(
        ["knn", "mnb", "tree"],
        weights=[0.22, 0.53, 0.25],
        k=1,
    )[0]
    if nome == "knn":
        return nome, KNeighborsClassifier(
            n_neighbors=rng.choice([3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17, 19, 21]),
            weights=rng.choice(["uniform", "distance"]),
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
    if nome == "mnb":
        return nome, MultinomialNB(
            alpha=rng.choice(
                [
                    0.001,
                    0.003,
                    0.005,
                    0.008,
                    0.01,
                    0.02,
                    0.04,
                    0.06,
                    0.08,
                    0.1,
                    0.12,
                    0.14,
                    0.18,
                    0.22,
                    0.28,
                    0.35,
                    0.45,
                    0.55,
                    0.7,
                    0.85,
                    1.0,
                    1.25,
                    1.5,
                ]
            )
        )
    return nome, DecisionTreeClassifier(
        max_depth=rng.choice([6, 8, 10, 12, 15, 18, 22, 28, 35, 45, None]),
        min_samples_split=rng.choice([2, 4, 5, 8, 10, 15]),
        min_samples_leaf=rng.choice([1, 2, 3, 4, 6]),
        max_leaf_nodes=rng.choice([None, None, None, 200, 500, 1000, 2000]),
        criterion=rng.choice(["gini", "entropy"]),
        random_state=RANDOM_STATE,
    )


def avaliar_pipeline(pipe: Pipeline, X, y, modo: str, n_splits: int):
    if modo == "holdout":
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        pipe.fit(X_tr, y_tr)
        return accuracy_score(y_va, pipe.predict(X_va))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr_idx, va_idx in cv.split(X, y):
        p = clone(pipe)
        p.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        scores.append(accuracy_score(y.iloc[va_idx], p.predict(X.iloc[va_idx])))
    return sum(scores) / len(scores)


def processar_tentativa(
    trial_id: int,
    modo: str,
    n_splits: int,
    vnome: str,
    vet,
    cnome: str,
    clf,
    X,
    y,
    test_df,
    out_dir: Path,
    historico: list,
    meta_local: float | None,
) -> Tuple[float, bool]:
    pipe = Pipeline([("vet", clone(vet)), ("clf", clone(clf))])

    try:
        acc_proxy = avaliar_pipeline(pipe, X, y, modo=modo, n_splits=n_splits)
    except Exception as exc:
        print(f"trial {trial_id:04d} ignorado (erro no fit): {exc}")
        return -1.0, False

    pipe_full = clone(pipe)
    pipe_full.fit(X, y)
    preds = pipe_full.predict(test_df["review_limpo"])

    fname = f"submission_t{trial_id:04d}_{modo}_proxy{acc_proxy:.4f}_{vnome}_{cnome}.csv"
    pd.DataFrame({"id": test_df.index, "label": preds}).to_csv(out_dir / fname, index=False)

    historico.append(
        {
            "arquivo": fname,
            "acuracia_proxy": acc_proxy,
            "modo_proxy": modo,
            "vetorizador": vnome,
            "classificador": cnome,
            "hiperparams": json.dumps({"vet": vnome, "clf": cnome, "pipe": str(pipe)}),
        }
    )
    return acc_proxy, meta_local is not None and acc_proxy >= meta_local


def main():
    global RANDOM_STATE
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trials",
        type=int,
        default=800,
        help="tentativas aleatórias após a grade (padrão alto para varrer mais)",
    )
    parser.add_argument(
        "--modo",
        choices=["holdout", "cv"],
        default="holdout",
        help="proxy: holdout 80/20 ou média em folds",
    )
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--meta-local", type=float, default=None)
    parser.add_argument("--copiar-melhor-para", type=str, default=None)
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="copia os k melhores do manifest para submission_rank01.csv ... na pasta de saída",
    )
    parser.add_argument(
        "--sem-grade",
        action="store_true",
        help="não roda a grade prioritária fixa (só amostragem aleatória)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="submissoes_candidatas",
        help="pasta para csv e manifest (use outra para não misturar com buscas antigas)",
    )
    args = parser.parse_args()

    RANDOM_STATE = args.seed
    rng = random.Random(RANDOM_STATE)

    data_dir = Path("aprendizado-de-maquina-26-1-competicao-2-pln")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv", index_col="id")
    test_df = pd.read_csv(data_dir / "test.csv", index_col="id")
    train_df["review_limpo"] = train_df["review"].map(limpar_texto)
    test_df["review_limpo"] = test_df["review"].map(limpar_texto)

    X = train_df["review_limpo"]
    y = train_df["label"]

    historico: list = []
    melhor_proxy = -1.0
    trial_id = 0
    parar = False

    if not args.sem_grade:
        print("fase 1: grade prioritária (combinações fixas)...")
        for vnome, vet, cnome, clf in iter_grade_prioritaria():
            acc, stop = processar_tentativa(
                trial_id,
                args.modo,
                args.cv_splits,
                vnome,
                vet,
                cnome,
                clf,
                X,
                y,
                test_df,
                out_dir,
                historico,
                args.meta_local,
            )
            if acc > melhor_proxy:
                melhor_proxy = acc
                print(f"novo melhor proxy {acc:.4f} -> submission_t{trial_id:04d}_...")
            trial_id += 1
            if stop:
                parar = True
                break
        print(f"fase 1 concluída ({trial_id} arquivos). melhor proxy até agora: {melhor_proxy:.4f}")

    if not parar:
        print(f"fase 2: amostragem aleatória ({args.trials} tentativas)...")
        for _ in range(args.trials):
            vnome, vet = amostrar_vetorizador(rng)
            cnome, clf = amostrar_classificador(rng)
            acc, stop = processar_tentativa(
                trial_id,
                args.modo,
                args.cv_splits,
                vnome,
                vet,
                cnome,
                clf,
                X,
                y,
                test_df,
                out_dir,
                historico,
                args.meta_local,
            )
            if acc > melhor_proxy:
                melhor_proxy = acc
                print(f"novo melhor proxy {acc:.4f} -> (trial {trial_id:04d})")
            trial_id += 1
            if stop:
                print(
                    f"parada: proxy {acc:.4f} >= {args.meta_local} "
                    "(confirme no kaggle; não é o score oficial)."
                )
                break

    manifest = pd.DataFrame(historico).sort_values("acuracia_proxy", ascending=False)
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    if args.copiar_melhor_para and len(manifest) > 0:
        melhor_arq = manifest.iloc[0]["arquivo"]
        dest = Path(args.copiar_melhor_para)
        shutil.copy(out_dir / melhor_arq, dest)
        print(f"cópia do melhor proxy -> {dest.resolve()}")

    if args.top_k > 0 and len(manifest) > 0:
        sub = manifest.head(args.top_k).reset_index(drop=True)
        for i, row in sub.iterrows():
            rank = i + 1
            alvo = out_dir / f"submission_rank{rank:02d}_proxy{row['acuracia_proxy']:.4f}.csv"
            shutil.copy(out_dir / row["arquivo"], alvo)
            print(f"top {rank} -> {alvo.name}")

    print("\n--- resumo ---")
    print(f"total de csv: {len(manifest)}")
    print(f"melhor acuracia_proxy ({args.modo}): {melhor_proxy:.4f}")
    print(
        "0,87 no pdf é no kaggle. com só knn/nb/árvore, o proxy local dificilmente chega a 0,87; "
        "submeta os primeiros do manifest e veja o leaderboard."
    )


RANDOM_STATE = 42

if __name__ == "__main__":
    main()
