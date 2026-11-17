"""Carregamento e validação da base autoral de exemplos de apelidos."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .nickname_normalization import normalize_nickname


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "curated" / "nickname_examples.csv"
REQUIRED_COLUMNS = (
    "nickname",
    "label",
    "source",
    "generation_rule",
    "root_group",
    "label_reason",
    "review_status",
)
VALID_LABELS = {"permitido", "inadequado", "ambiguo"}
VALID_SOURCES = {"ldnoobw", "manual", "sintetico"}
VALID_REVIEW_STATUSES = {"pendente", "revisado"}


def load_dataset(path: str | Path = DEFAULT_DATASET) -> tuple[list[str], list[dict[str, str]]]:
    """Carrega o CSV sem modificar os valores originais."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as dataset_file:
        reader = csv.DictReader(dataset_file)
        columns = reader.fieldnames or []
        rows = [
            {key: value if value is not None else "" for key, value in row.items()}
            for row in reader
        ]
    return columns, rows


def _duplicate_groups(
    rows: list[dict[str, str]], key_function: Any
) -> list[dict[str, Any]]:
    grouped_rows: dict[str, list[int]] = defaultdict(list)
    for line_number, row in enumerate(rows, start=2):
        grouped_rows[key_function(row.get("nickname", ""))].append(line_number)
    return [
        {"nickname": rows[line_numbers[0] - 2].get("nickname", ""), "lines": line_numbers}
        for key, line_numbers in grouped_rows.items()
        if key and len(line_numbers) > 1
    ]


def validate_dataset(
    columns: list[str], rows: list[dict[str, str]]
) -> dict[str, Any]:
    """Retorna erros estruturais e duplicatas sem alterar as linhas."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in columns]
    errors: list[dict[str, Any]] = []

    for line_number, row in enumerate(rows, start=2):
        if not row.get("nickname", "").strip():
            errors.append({"line": line_number, "field": "nickname", "issue": "vazio"})
        label = row.get("label", "")
        pending_without_suggestion = (
            row.get("review_status") == "pendente" and not label
        )
        if label not in VALID_LABELS and not pending_without_suggestion:
            errors.append({"line": line_number, "field": "label", "issue": "rotulo_invalido"})
        if row.get("source", "") not in VALID_SOURCES:
            errors.append({"line": line_number, "field": "source", "issue": "origem_invalida"})
        if row.get("review_status", "") not in VALID_REVIEW_STATUSES:
            errors.append(
                {"line": line_number, "field": "review_status", "issue": "status_invalido"}
            )
        if not row.get("label_reason", "").strip():
            errors.append(
                {"line": line_number, "field": "label_reason", "issue": "justificativa_ausente"}
            )
        if not row.get("root_group", "").strip():
            errors.append(
                {"line": line_number, "field": "root_group", "issue": "grupo_ausente"}
            )
        if (
            row.get("source") == "sintetico"
            and not row.get("generation_rule", "").strip()
        ):
            errors.append(
                {"line": line_number, "field": "generation_rule", "issue": "regra_ausente"}
            )

    exact_duplicates = _duplicate_groups(rows, lambda value: value)
    normalized_duplicates = _duplicate_groups(
        rows, lambda value: normalize_nickname(value)
    )
    return {
        "missing_columns": missing_columns,
        "errors": errors,
        "exact_duplicates": exact_duplicates,
        "normalized_duplicates": normalized_duplicates,
    }


def summarize_dataset(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Resume contagens por rótulo e estado de revisão."""
    labels = Counter(row.get("label", "") for row in rows)
    review_statuses = Counter(row.get("review_status", "") for row in rows)
    return {
        "total_examples": len(rows),
        "by_label": {
            label: labels.get(label, 0)
            for label in sorted(VALID_LABELS)
        },
        "without_label": labels.get("", 0),
        "by_review_status": {
            status: review_statuses.get(status, 0)
            for status in sorted(VALID_REVIEW_STATUSES)
        },
    }


def inspect_dataset(path: str | Path = DEFAULT_DATASET) -> dict[str, Any]:
    """Carrega, valida e resume um CSV de exemplos."""
    columns, rows = load_dataset(path)
    return {
        "path": str(Path(path)),
        "summary": summarize_dataset(rows),
        "validation": validate_dataset(columns, rows),
    }


def split_by_root_group(
    rows: list[dict[str, str]],
    test_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Separa grupos inteiros; não é executada automaticamente nesta etapa."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction deve estar entre 0 e 1.")
    if any(not row.get("root_group", "").strip() for row in rows):
        raise ValueError("Todos os exemplos precisam de root_group para a divisão.")

    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped_rows[row["root_group"]].append(row)

    groups = list(grouped_rows)
    if rows and len(groups) < 2:
        raise ValueError("A divisão exige pelo menos dois valores distintos de root_group.")
    random.Random(seed).shuffle(groups)
    target_test_size = round(len(rows) * test_fraction)
    test_groups: set[str] = set()
    test_size = 0
    for group in groups:
        if abs((test_size + len(grouped_rows[group])) - target_test_size) < abs(
            test_size - target_test_size
        ):
            test_groups.add(group)
            test_size += len(grouped_rows[group])
    if rows and not test_groups:
        smallest_group = min(groups, key=lambda group: len(grouped_rows[group]))
        test_groups.add(smallest_group)
    if groups and len(test_groups) == len(groups):
        smallest_group = min(test_groups, key=lambda group: len(grouped_rows[group]))
        test_groups.remove(smallest_group)

    train_rows = [
        row for row in rows if row["root_group"] not in test_groups
    ]
    test_rows = [row for row in rows if row["root_group"] in test_groups]
    return train_rows, test_rows


def main() -> int:
    report = inspect_dataset()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    validation = report["validation"]
    return int(
        bool(
            validation["missing_columns"]
            or validation["errors"]
            or validation["exact_duplicates"]
            or validation["normalized_duplicates"]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
