"""Preparação, avaliação e previsão experimental de apelidos.

As dependências pandas, scikit-learn e joblib são importadas apenas quando
um treino ou uma previsão com modelo efetivamente solicitado é possível.
"""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping, Sequence

from .lexical_filter import (
    CandidateReview,
    LexicalCandidate,
    check_nickname,
    load_candidate_reviews,
)
from .nickname_dataset import (
    DEFAULT_DATASET,
    VALID_LABELS,
    load_dataset,
    validate_dataset,
)
from .nickname_normalization import normalize_nickname


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "nickname_moderation.joblib"
DEFAULT_MANUAL_EVALUATION_PATH = (
    PROJECT_ROOT / "data" / "curated" / "manual_evaluation.csv"
)
DEFAULT_MODERATION_DATASET_PATH = (
    PROJECT_ROOT / "data" / "curated" / "bytecraft_nicknames_moderacao.csv"
)
TARGET_LABELS = {"permitido", "inadequado"}
MODEL_LABELS = ("permitido", "inadequado")


@dataclass(frozen=True)
class TrainingConfig:
    """Parâmetros de treinamento e decisão para avaliação agrupada."""

    test_fraction: float = 0.2
    random_state: int = 42
    minimum_groups_per_class: int = 5
    minimum_examples_per_class: int = 5
    minimum_confidence: float = 0.6
    split_attempts: int = 128


def _count_distinct_groups_by_label(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, int]:
    return {
        label: len(
            {
                row.get("root_group", "").strip()
                for row in rows
                if row.get("label") == label and row.get("root_group", "").strip()
            }
        )
        for label in sorted(TARGET_LABELS)
    }


def _root_group_label_conflicts(
    rows: Sequence[Mapping[str, str]],
    labels: set[str] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, set[str]] = {}
    for row in rows:
        label = row.get("label", "")
        group = row.get("root_group", "").strip()
        if group and (labels is None or label in labels):
            grouped.setdefault(group, set()).add(label)
    return [
        {"root_group": group, "labels": sorted(group_labels)}
        for group, group_labels in sorted(grouped.items())
        if len(group_labels) > 1
    ]


def audit_training_data(
    path: str | Path = DEFAULT_DATASET,
    config: TrainingConfig = TrainingConfig(),
) -> dict[str, Any]:
    """Audita rótulos e revisão; não presume que rótulo preenchido foi validado."""
    columns, rows = load_dataset(path)
    validation = validate_dataset(columns, rows)
    status_counts = Counter(row.get("review_status", "") for row in rows)
    all_label_counts = Counter(row.get("label", "") for row in rows)
    reviewed_rows = [row for row in rows if row.get("review_status") == "revisado"]
    eligible_rows = [
        row
        for row in reviewed_rows
        if row.get("label") in TARGET_LABELS
        and row.get("nickname", "").strip()
        and row.get("root_group", "").strip()
        and row.get("label_reason", "").strip()
    ]
    ambiguous_count = sum(row.get("label") == "ambiguo" for row in rows)
    reviewed_ambiguous_count = sum(
        row.get("label") == "ambiguo" for row in reviewed_rows
    )
    missing_group_count = sum(not row.get("root_group", "").strip() for row in reviewed_rows)
    missing_reason_count = sum(not row.get("label_reason", "").strip() for row in reviewed_rows)
    reviewed_target_counts = Counter(
        row.get("label", "") for row in reviewed_rows if row.get("label") in TARGET_LABELS
    )
    eligible_counts = Counter(row["label"] for row in eligible_rows)
    distinct_groups = _count_distinct_groups_by_label(eligible_rows)
    blockers = audit_training_readiness(reviewed_rows, config)
    if validation["missing_columns"] or validation["errors"]:
        blockers.append("dataset_validation_failed")
    if validation["exact_duplicates"] or validation["normalized_duplicates"]:
        blockers.append("dataset_duplicates_found")
    target_conflicts = _root_group_label_conflicts(reviewed_rows, TARGET_LABELS)
    if target_conflicts:
        blockers.append("root_group_target_label_conflict")
    return {
        "total_examples": len(rows),
        "label_counts_all": {
            label: all_label_counts.get(label, 0) for label in sorted(VALID_LABELS)
        },
        "review_status_counts": {
            status: status_counts.get(status, 0)
            for status in ("pendente", "revisado")
        },
        "reviewed_examples": len(reviewed_rows),
        "reviewed_label_counts": {
            label: sum(
                row.get("review_status") == "revisado"
                and row.get("label") == label
                for row in rows
            )
            for label in sorted(VALID_LABELS)
        },
        "reviewed_target_counts": {
            label: reviewed_target_counts.get(label, 0)
            for label in MODEL_LABELS
        },
        "root_groups_all_by_class": _count_distinct_groups_by_label(rows),
        "root_groups_reviewed_by_class": _count_distinct_groups_by_label(reviewed_rows),
        "reviewed_ambiguous_excluded": reviewed_ambiguous_count,
        "ambiguous_examples_total": ambiguous_count,
        "eligible_examples": len(eligible_rows),
        "eligible_counts": {
            label: eligible_counts.get(label, 0) for label in MODEL_LABELS
        },
        "eligible_root_groups_by_class": distinct_groups,
        "root_group_label_conflicts": _root_group_label_conflicts(reviewed_rows),
        "root_group_target_label_conflicts": target_conflicts,
        "dataset_validation": validation,
        "exclusion_counts": {
            "pending_review": status_counts.get("pendente", 0),
            "reviewed_ambiguous_label": reviewed_ambiguous_count,
            "missing_root_group_among_reviewed": missing_group_count,
            "missing_reason_among_reviewed": missing_reason_count,
            "reviewed_non_binary_label": sum(
                row.get("label") not in TARGET_LABELS
                for row in reviewed_rows
            ),
        },
        "training_ready": not blockers,
        "training_blockers": blockers,
    }


def eligible_rows(path: str | Path = DEFAULT_DATASET) -> list[dict[str, str]]:
    """Seleciona apenas exemplos revisados, binários e rastreáveis."""
    _, rows = load_dataset(path)
    return [
        row
        for row in rows
        if row.get("review_status") == "revisado"
        and row.get("label") in TARGET_LABELS
        and row.get("nickname", "").strip()
        and row.get("root_group", "").strip()
        and row.get("label_reason", "").strip()
    ]


def _load_moderation_cases(path: str | Path) -> list[dict[str, str]]:
    """Carrega o CSV ByteCraft e colapsa variantes repetidas consistentes."""
    required_columns = {
        "nickname",
        "expected_label",
        "review_status",
        "notes",
        "root_group",
    }
    cases: list[dict[str, str]] = []
    seen: dict[str, tuple[str, str]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as dataset_file:
        reader = csv.DictReader(dataset_file)
        missing = required_columns.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Colunas ausentes em {Path(path).name}: {sorted(missing)}"
            )
        for line_number, raw_row in enumerate(reader, start=2):
            nickname = (raw_row.get("nickname") or "").strip()
            label = (raw_row.get("expected_label") or "").strip()
            review_status = (raw_row.get("review_status") or "").strip()
            notes = (raw_row.get("notes") or "").strip()
            root_group = (raw_row.get("root_group") or "").strip()
            if not nickname:
                raise ValueError(
                    f"Apelido vazio em {Path(path).name}, linha {line_number}."
                )
            if review_status not in {"pendente", "revisado"}:
                raise ValueError(
                    f"Estado de revisão inválido em {Path(path).name}, "
                    f"linha {line_number}."
                )
            if review_status == "pendente" and label:
                raise ValueError(
                    f"Rótulo pendente preenchido em {Path(path).name}, "
                    f"linha {line_number}."
                )
            if review_status == "revisado" and label not in VALID_LABELS:
                raise ValueError(
                    f"Rótulo inválido em {Path(path).name}, linha {line_number}."
                )
            if review_status == "revisado" and (not notes or not root_group):
                raise ValueError(
                    f"Justificativa e root_group são obrigatórios para casos "
                    f"revisados em {Path(path).name}, linha {line_number}."
                )
            normalized = normalize_nickname(nickname)
            existing = seen.get(normalized)
            if existing is not None:
                if existing != (label, root_group):
                    raise ValueError(
                        f"Duplicata conflitante após normalização em "
                        f"{Path(path).name}, linha {line_number}."
                    )
                continue
            seen[normalized] = (label, root_group)
            cases.append(
                {
                    "nickname": nickname,
                    "expected_label": label,
                    "review_status": review_status,
                    "notes": notes,
                    "root_group": root_group,
                }
            )
    return cases


def _count_csv_records(path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as dataset_file:
        return sum(1 for _ in csv.DictReader(dataset_file))


def load_training_records(
    dataset_path: str | Path = DEFAULT_DATASET,
    manual_evaluation_path: str | Path | None = DEFAULT_MANUAL_EVALUATION_PATH,
    moderation_dataset_path: str | Path | None = DEFAULT_MODERATION_DATASET_PATH,
) -> list[dict[str, str]]:
    """Combina fontes revisadas, sem duplicar apelidos normalizados."""
    records = prepare_training_records(eligible_rows(dataset_path))
    by_nickname = {
        row["normalized_nickname"]: row
        for row in records
    }

    from .evaluate_manual_cases import load_manual_cases

    if manual_evaluation_path is not None:
        sources = (
            (
                load_manual_cases(manual_evaluation_path),
                "manual_evaluation",
            ),
        )
    else:
        sources = ()
    if moderation_dataset_path is not None:
        sources += (
            (
                _load_moderation_cases(moderation_dataset_path),
                "bytecraft_nicknames_moderacao",
            ),
        )
    for cases, source_name in sources:
        for case in cases:
            if case["review_status"] != "revisado":
                continue
            label = case["expected_label"]
            if label not in TARGET_LABELS:
                continue
            root_group = case["root_group"]
            if not root_group:
                raise ValueError(
                    f"O caso {case['nickname']!r} em {source_name} "
                    "não tem root_group."
                )
            record = {
                "nickname": case["nickname"],
                "normalized_nickname": normalize_nickname(case["nickname"]),
                "label": label,
                "root_group": root_group,
                "source": source_name,
                "generation_rule": f"caso revisado em {source_name}",
                "label_reason": case["notes"],
                "review_status": "revisado",
            }
            existing = by_nickname.get(record["normalized_nickname"])
            if existing is not None:
                if existing["label"] != label:
                    raise ValueError(
                        f"Conflito de rótulo para {case['nickname']!r}: "
                        f"{existing['label']} na base e {label} em {source_name}."
                    )
                continue
            by_nickname[record["normalized_nickname"]] = record
            records.append(record)

    return records


def prepare_training_records(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Preserva apelido e metadados, adicionando texto normalizado à parte."""
    prepared: list[dict[str, str]] = []
    for row in rows:
        if (
            row.get("review_status") != "revisado"
            or row.get("label") not in TARGET_LABELS
        ):
            continue
        if not row.get("nickname", "").strip():
            continue
        if not row.get("root_group", "").strip():
            continue
        if not row.get("label_reason", "").strip():
            continue
        prepared.append(
            {
                "nickname": row["nickname"],
                "normalized_nickname": normalize_nickname(row["nickname"]),
                "label": row["label"],
                "root_group": row["root_group"],
                "source": row.get("source", ""),
                "generation_rule": row.get("generation_rule", ""),
                "label_reason": row["label_reason"],
                "review_status": row["review_status"],
            }
        )
    return prepared


def audit_training_readiness(
    rows: Sequence[Mapping[str, str]],
    config: TrainingConfig = TrainingConfig(),
) -> list[str]:
    """Retorna bloqueios objetivos antes de qualquer import de ML."""
    blockers: list[str] = []
    prepared = prepare_training_records(rows)
    if not prepared:
        blockers.append("nenhum_exemplo_revisado_elegivel")

    counts = Counter(row["label"] for row in prepared)
    groups = _count_distinct_groups_by_label(prepared)
    if _root_group_label_conflicts(prepared, TARGET_LABELS):
        blockers.append("root_group_target_label_conflict")
    for label in MODEL_LABELS:
        if counts.get(label, 0) < config.minimum_examples_per_class:
            blockers.append(f"exemplos_insuficientes:{label}")
        if groups[label] < config.minimum_groups_per_class:
            blockers.append(f"grupos_insuficientes:{label}")
    return blockers


def build_training_frame(rows: Sequence[Mapping[str, str]]) -> Any:
    """Cria DataFrame só com exemplos elegíveis, preservando campos de auditoria."""
    prepared = prepare_training_records(rows)
    if not prepared:
        raise ValueError("Não há exemplos revisados elegíveis para preparar.")
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError(
            "pandas é necessário para preparar o DataFrame de treinamento."
        ) from error
    return pd.DataFrame.from_records(prepared)


def build_model_pipeline() -> Any:
    """Cria pipeline TF-IDF de caracteres + regressão logística."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except ImportError as error:
        raise RuntimeError(
            "scikit-learn é necessário para construir o pipeline do modelo."
        ) from error
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    lowercase=False,
                    preprocessor=normalize_nickname,
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )


def split_records_by_root_group(
    rows: Sequence[Mapping[str, str]],
    config: TrainingConfig = TrainingConfig(),
) -> tuple[list[Mapping[str, str]], list[Mapping[str, str]]]:
    """Busca divisão agrupada válida; nenhuma família pode cruzar os conjuntos."""
    prepared = prepare_training_records(rows)
    blockers = audit_training_readiness(prepared, config)
    if blockers:
        raise ValueError(
            "Dados insuficientes para divisão de avaliação: " + ", ".join(blockers)
        )
    try:
        from sklearn.model_selection import GroupShuffleSplit
    except ImportError as error:
        raise RuntimeError(
            "scikit-learn é necessário para separar os grupos de avaliação."
        ) from error

    groups = [row["root_group"] for row in prepared]
    labels = [row["label"] for row in prepared]
    best_split: tuple[list[Mapping[str, str]], list[Mapping[str, str]]] | None = None
    best_distance: float | None = None
    splitter = GroupShuffleSplit(
        n_splits=config.split_attempts,
        test_size=config.test_fraction,
        random_state=config.random_state,
    )
    for train_indices, test_indices in splitter.split(prepared, labels, groups):
        train_rows = [prepared[index] for index in train_indices]
        test_rows = [prepared[index] for index in test_indices]
        if {row["label"] for row in train_rows} != set(MODEL_LABELS):
            continue
        if {row["label"] for row in test_rows} != set(MODEL_LABELS):
            continue
        train_groups = {row["root_group"] for row in train_rows}
        test_groups = {row["root_group"] for row in test_rows}
        if train_groups & test_groups:
            continue
        distance = abs(
            len(test_rows) / len(prepared) - config.test_fraction
        )
        if best_distance is None or distance < best_distance:
            best_split = train_rows, test_rows
            best_distance = distance

    if best_split is None:
        raise ValueError(
            "Não foi possível formar conjuntos de treino e teste com as duas "
            "classes e sem separar root_group."
        )
    return best_split


def _lexical_baseline_label(result: Mapping[str, Any]) -> str | None:
    status = result.get("status")
    if status == "inadequado_claro":
        return "inadequado"
    if status == "neutro" and any(
        match.get("classification") == "neutro"
        for match in result.get("matches", [])
    ):
        return "permitido"
    return None


def evaluate_model(
    model: Any,
    test_rows: Sequence[Mapping[str, str]],
    *,
    minimum_confidence: float = TrainingConfig.minimum_confidence,
    candidate_terms: list[LexicalCandidate] | None = None,
    candidate_reviews: Mapping[int, CandidateReview] | None = None,
) -> dict[str, Any]:
    """Avalia holdout por grupos e compara a linha de base lexical revisada."""
    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            precision_recall_fscore_support,
        )
    except ImportError as error:
        raise RuntimeError("scikit-learn é necessário para avaliar o modelo.") from error

    nicknames = [row["nickname"] for row in test_rows]
    actual = [row["label"] for row in test_rows]
    predicted = list(model.predict(nicknames))
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise RuntimeError(
            "O modelo precisa fornecer classes_ e predict_proba para avaliar "
            "bloqueios automáticos."
        )
    probabilities = model.predict_proba(nicknames)
    class_labels = [str(label) for label in model.classes_]
    inappropriate_index = class_labels.index("inadequado")
    inappropriate_scores = [
        float(probability[inappropriate_index]) for probability in probabilities
    ]
    automatically_blocked = [
        score >= minimum_confidence for score in inappropriate_scores
    ]
    automatic_true_positives = sum(
        blocked and truth == "inadequado"
        for blocked, truth in zip(automatically_blocked, actual)
    )
    automatic_false_positives = sum(
        blocked and truth == "permitido"
        for blocked, truth in zip(automatically_blocked, actual)
    )
    automatic_false_negatives = sum(
        not blocked and truth == "inadequado"
        for blocked, truth in zip(automatically_blocked, actual)
    )
    automatic_true_negatives = sum(
        not blocked and truth == "permitido"
        for blocked, truth in zip(automatically_blocked, actual)
    )
    automatic_blocked_count = sum(automatically_blocked)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        labels=["inadequado"],
        average="binary",
        pos_label="inadequado",
        zero_division=0,
    )
    false_positives = sum(
        truth == "permitido" and guess == "inadequado"
        for truth, guess in zip(actual, predicted)
    )
    false_negatives = sum(
        truth == "inadequado" and guess == "permitido"
        for truth, guess in zip(actual, predicted)
    )
    per_class_report = classification_report(
        actual,
        predicted,
        labels=list(MODEL_LABELS),
        output_dict=True,
        zero_division=0,
    )

    baseline_actual: list[str] = []
    baseline_predicted: list[str] = []
    baseline_skipped = 0
    baseline_review_ledger = (
        candidate_reviews
        if candidate_reviews is not None
        else load_candidate_reviews()
    )
    for row in test_rows:
        lexical = check_nickname(
            row["nickname"],
            candidates=candidate_terms,
            reviews=baseline_review_ledger,
        )
        lexical_label = _lexical_baseline_label(lexical)
        if lexical_label is None:
            baseline_skipped += 1
            continue
        baseline_actual.append(row["label"])
        baseline_predicted.append(lexical_label)

    baseline_metrics: dict[str, Any] | None = None
    if baseline_actual and set(baseline_actual) == set(MODEL_LABELS):
        baseline_precision, baseline_recall, baseline_f1, _ = (
            precision_recall_fscore_support(
                baseline_actual,
                baseline_predicted,
                labels=["inadequado"],
                average="binary",
                pos_label="inadequado",
                zero_division=0,
            )
        )
        baseline_metrics = {
            "precision_inadequado": float(baseline_precision),
            "recall_inadequado": float(baseline_recall),
            "f1_inadequado": float(baseline_f1),
            "confusion_matrix": confusion_matrix(
                baseline_actual,
                baseline_predicted,
                labels=list(MODEL_LABELS),
            ).tolist(),
            "evaluated_examples": len(baseline_actual),
        }

    return {
        "evaluated_examples": len(test_rows),
        "accuracy": float(accuracy_score(actual, predicted)),
        "per_class": {
            label: {
                metric: float(per_class_report[label][metric])
                for metric in ("precision", "recall", "f1-score", "support")
            }
            for label in MODEL_LABELS
        },
        "macro_f1": float(per_class_report["macro avg"]["f1-score"]),
        "precision_inadequado": float(precision),
        "recall_inadequado": float(recall),
        "f1_inadequado": float(f1),
        "confusion_matrix_labels": list(MODEL_LABELS),
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=list(MODEL_LABELS)
        ).tolist(),
        "false_positives_count": false_positives,
        "false_negatives_count": false_negatives,
        "automatic_block_decision": {
            "minimum_confidence": minimum_confidence,
            "blocked_examples": automatic_blocked_count,
            "true_positives": automatic_true_positives,
            "false_positives": automatic_false_positives,
            "false_negatives": automatic_false_negatives,
            "true_negatives": automatic_true_negatives,
            "precision": (
                automatic_true_positives / automatic_blocked_count
                if automatic_blocked_count
                else 0.0
            ),
            "recall": (
                automatic_true_positives
                / (automatic_true_positives + automatic_false_negatives)
                if automatic_true_positives + automatic_false_negatives
                else 0.0
            ),
        },
        "lexical_baseline": baseline_metrics,
        "lexical_baseline_skipped_uncertain": baseline_skipped,
        "lexical_baseline_coverage": (
            len(baseline_actual) / len(test_rows) if test_rows else 0.0
        ),
    }


def run_training(
    dataset_path: str | Path = DEFAULT_DATASET,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    config: TrainingConfig = TrainingConfig(),
    *,
    manual_evaluation_path: str | Path | None = DEFAULT_MANUAL_EVALUATION_PATH,
    moderation_dataset_path: str | Path | None = DEFAULT_MODERATION_DATASET_PATH,
) -> dict[str, Any]:
    """Treina e salva apenas se dados revisados e divisão por grupos forem suficientes."""
    _, rows = load_dataset(dataset_path)
    prepared = load_training_records(
        dataset_path,
        manual_evaluation_path,
        moderation_dataset_path,
    )
    audit = audit_training_data(dataset_path, config)
    blockers = list(audit["training_blockers"])
    blockers.extend(audit_training_readiness(prepared, config))
    blockers = list(dict.fromkeys(blockers))
    source_hashes: dict[str, str] = {}
    source_record_counts = {Path(dataset_path).name: len(rows)}
    for source_path in (manual_evaluation_path, moderation_dataset_path):
        if source_path is None:
            continue
        source_hashes[Path(source_path).name] = hashlib.sha256(
            Path(source_path).read_bytes()
        ).hexdigest()
        source_record_counts[Path(source_path).name] = _count_csv_records(source_path)
    report: dict[str, Any] = {
        "training_executed": False,
        "model_path": None,
        "metadata_path": None,
        "eligible_examples": len(prepared),
        "eligible_counts": {
            label: sum(row["label"] == label for row in prepared)
            for label in MODEL_LABELS
        },
        "eligible_root_groups_by_class": _count_distinct_groups_by_label(prepared),
        "excluded_ambiguous": audit["exclusion_counts"]["reviewed_ambiguous_label"],
        "examples_included_by_source": dict(
            Counter(row["source"] for row in prepared)
        ),
        "blockers": blockers,
    }
    if blockers:
        return report

    train_rows, test_rows = split_records_by_root_group(prepared, config)
    frame = build_training_frame(train_rows)
    pipeline = build_model_pipeline()
    pipeline.fit(frame["nickname"], frame["label"])
    metrics = evaluate_model(
        pipeline,
        test_rows,
        minimum_confidence=config.minimum_confidence,
    )

    final_frame = build_training_frame(prepared)
    final_pipeline = build_model_pipeline()
    final_pipeline.fit(final_frame["nickname"], final_frame["label"])
    try:
        import joblib
        import pandas
        import sklearn
    except ImportError as error:
        raise RuntimeError(
            "pandas, scikit-learn e joblib são necessários para treinar e salvar."
        ) from error

    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = destination.with_suffix(".metadata.json")
    source_hashes[Path(dataset_path).name] = hashlib.sha256(
        Path(dataset_path).read_bytes()
    ).hexdigest()
    dataset_digest = hashlib.sha256()
    dataset_digest.update(Path(dataset_path).name.encode("utf-8"))
    dataset_digest.update(Path(dataset_path).read_bytes())
    for source_path in (manual_evaluation_path, moderation_dataset_path):
        if source_path is None:
            continue
        dataset_digest.update(Path(source_path).name.encode("utf-8"))
        dataset_digest.update(Path(source_path).read_bytes())
    metadata = {
        "model_version": "1.3.0-experimental",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": dataset_digest.hexdigest(),
        "dataset_sources_sha256": source_hashes,
        "dataset_examples_total": sum(source_record_counts.values()),
        "dataset_examples_by_source": source_record_counts,
        "training_examples_by_source": dict(
            Counter(row["source"] for row in prepared)
        ),
        "training_examples": len(prepared),
        "training_classes": list(MODEL_LABELS),
        "training_class_counts": dict(Counter(row["label"] for row in prepared)),
        "training_groups": len({row["root_group"] for row in prepared}),
        "excluded_ambiguous": audit["exclusion_counts"]["reviewed_ambiguous_label"],
        "split": {
            "strategy": "GroupShuffleSplit por root_group",
            "test_fraction": config.test_fraction,
            "random_state": config.random_state,
            "train_examples": len(train_rows),
            "test_examples": len(test_rows),
            "train_groups": len({row["root_group"] for row in train_rows}),
            "test_groups": len({row["root_group"] for row in test_rows}),
        },
        "pipeline": {
            "vectorizer": {
                "class": "TfidfVectorizer",
                "analyzer": "char",
                "ngram_range": [2, 5],
                "preprocessor": "src.nickname_normalization.normalize_nickname",
            },
            "classifier": {
                "class": "LogisticRegression",
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": 42,
                "solver": "liblinear",
            },
        },
        "evaluation": metrics,
        "versions": {
            "pandas": pandas.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": version("joblib"),
        },
        "limitations": [
            "Base sintética e não representativa de apelidos reais de estudantes.",
            "Métricas exploratórias de um único holdout agrupado.",
            "O limiar 0.60 foi escolhido experimentalmente e ainda apresenta falsos positivos.",
            "Rótulo ambiguo excluído do treinamento supervisionado.",
        ],
    }

    model_temp = _temporary_path(destination)
    metadata_temp = _temporary_path(metadata_path)
    try:
        joblib.dump(final_pipeline, model_temp)
        with metadata_temp.open("w", encoding="utf-8", newline="\n") as metadata_file:
            json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
            metadata_file.write("\n")
            metadata_file.flush()
            os.fsync(metadata_file.fileno())
        os.replace(model_temp, destination)
        os.replace(metadata_temp, metadata_path)
    except Exception:
        model_temp.unlink(missing_ok=True)
        metadata_temp.unlink(missing_ok=True)
        raise

    report.update(
        {
            "training_executed": True,
            "model_path": str(destination),
            "metadata_path": str(metadata_path),
            "metrics": metrics,
            "train_examples": len(train_rows),
            "test_examples": len(test_rows),
            "train_groups": len({row["root_group"] for row in train_rows}),
            "test_groups": len({row["root_group"] for row in test_rows}),
        }
    )
    return report


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        return Path(temporary_file.name)


def load_saved_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Any:
    """Carrega um pipeline salvo sem iniciar novo treinamento."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        import joblib
    except ImportError as error:
        raise RuntimeError("joblib é necessário para carregar o modelo.") from error
    return joblib.load(path)


def predict_nickname(
    nickname: str,
    model: Any | None = None,
    *,
    model_path: str | Path | None = None,
    minimum_confidence: float = TrainingConfig.minimum_confidence,
    candidate_terms: list[LexicalCandidate] | None = None,
    candidate_reviews: Mapping[int, CandidateReview] | None = None,
) -> dict[str, Any]:
    """Prediz sem apresentar probabilidade como garantia de segurança."""
    if not nickname.strip():
        raise ValueError("O apelido não pode ficar vazio.")
    active_model = model
    if active_model is None and model_path is not None:
        active_model = load_saved_model(model_path)

    _, curated_rows = load_dataset()
    curated_entries = [
        (index, row)
        for index, row in enumerate(curated_rows, start=1)
        if row.get("label") == "inadequado"
        and row.get("review_status") == "revisado"
    ]
    curated_candidates = [
        LexicalCandidate(
            source_line=index,
            term=row["nickname"],
            normalized_term=normalize_nickname(row["nickname"]),
        )
        for index, row in curated_entries
    ]
    curated_reviews = {
        index: CandidateReview("inadequado_claro", row["label_reason"])
        for index, row in curated_entries
    }
    from .evaluate_manual_cases import load_manual_cases

    manual_cases = load_manual_cases()
    lexical_case_sources = (
        (manual_cases, len(curated_rows) + 1),
        (
            _load_moderation_cases(DEFAULT_MODERATION_DATASET_PATH),
            len(curated_rows) + len(manual_cases) + 1,
        ),
    )
    for cases, first_index in lexical_case_sources:
        for index, case in enumerate(cases, start=first_index):
            if (
                case["review_status"] != "revisado"
                or case["expected_label"] != "inadequado"
            ):
                continue
            curated_candidates.append(
                LexicalCandidate(
                    source_line=index,
                    term=case["nickname"],
                    normalized_term=normalize_nickname(case["nickname"]),
                )
            )
            curated_reviews[index] = CandidateReview(
                "inadequado_claro",
                case["notes"],
            )
    curated_lexical = check_nickname(
        nickname,
        candidates=curated_candidates,
        reviews=curated_reviews,
    )
    automatic_block = curated_lexical["status"] == "inadequado_claro"

    reviews = candidate_reviews
    if reviews is None:
        reviews = load_candidate_reviews() if candidate_terms is None else {}
    lexical = check_nickname(
        nickname,
        candidates=candidate_terms,
        reviews=reviews,
    )
    if active_model is None:
        return {
            "nickname_original": nickname,
            "predicted_class": None,
            "probabilities": None,
            "lexical_result": lexical,
            "uncertain": True,
            "uncertainty_reason": "modelo_nao_disponivel",
            "reviewed_lexical_result": curated_lexical,
            "automatic_block": automatic_block,
        }

    predicted = str(active_model.predict([nickname])[0])
    probabilities: dict[str, float] | None = None
    if hasattr(active_model, "predict_proba"):
        raw_probabilities = active_model.predict_proba([nickname])[0]
        probabilities = {
            str(label): float(probability)
            for label, probability in zip(active_model.classes_, raw_probabilities)
        }
    confidence = max(probabilities.values()) if probabilities else None
    uncertain_reasons: list[str] = []
    if confidence is None:
        uncertain_reasons.append("probabilidade_indisponivel")
    elif confidence < minimum_confidence:
        uncertain_reasons.append("confianca_abaixo_do_limite")
    if lexical["status"] in {"ambiguo", "pendente_revisao"}:
        uncertain_reasons.append("correspondencia_lexical_requer_revisao")
    lexical_label = _lexical_baseline_label(lexical)
    if lexical_label is not None and lexical_label != predicted:
        uncertain_reasons.append("discordancia_com_regra_lexical_revisada")

    return {
        "nickname_original": nickname,
        "predicted_class": predicted,
        "probabilities": probabilities,
        "lexical_result": lexical,
        "uncertain": bool(uncertain_reasons),
        "uncertainty_reason": uncertain_reasons,
        "reviewed_lexical_result": curated_lexical,
        "automatic_block": automatic_block,
        "probabilities_are_not_guarantees": True,
    }


def main() -> int:
    audit = audit_training_data()
    report = run_training()
    audit["training_result"] = report
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if report["training_executed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
