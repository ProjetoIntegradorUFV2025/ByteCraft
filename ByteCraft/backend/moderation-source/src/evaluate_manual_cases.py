"""Gera diagnósticos do conjunto separado de avaliação manual pendente."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .lexical_filter import CandidateReview, LexicalCandidate
from .nickname_dataset import DEFAULT_DATASET, VALID_LABELS, load_dataset
from .nickname_model import (
    DEFAULT_MODEL_PATH,
    eligible_rows,
    load_saved_model,
    predict_nickname,
    split_records_by_root_group,
)
from .nickname_normalization import normalize_nickname


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = (
    PROJECT_ROOT / "data" / "curated" / "manual_evaluation.csv"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "manual_evaluation_diagnostic.csv"
)
REQUIRED_COLUMNS = {
    "nickname",
    "expected_label",
    "review_status",
    "notes",
    "root_group",
}
REASON_MESSAGES = {
    "confianca_abaixo_do_limite": "confiança abaixo de 0,60",
    "probabilidade_indisponivel": "probabilidade indisponível",
    "correspondencia_lexical_requer_revisao": (
        "correspondência lexical pendente ou ambígua"
    ),
    "discordancia_com_regra_lexical_revisada": (
        "divergência com uma regra lexical revisada"
    ),
    "modelo_nao_disponivel": "modelo indisponível",
}
RESULT_MESSAGES = {
    "permitido": "APARENTEMENTE ADEQUADO",
    "inadequado": "INADEQUADO",
    "ambiguo": "REVISÃO NECESSÁRIA",
}
REPORT_COLUMNS = (
    "nickname",
    "model_prediction",
    "probability_permitido",
    "probability_inadequado",
    "lexical_status",
    "lexical_matches",
    "final_result",
    "review_reason",
    "base_presence",
    "base_match_nickname",
    "base_root_group",
    "expected_label",
    "review_status",
    "notes",
)


def load_manual_cases(
    path: str | Path = DEFAULT_CASES_PATH,
) -> list[dict[str, str]]:
    """Carrega casos; referência só pode ser preenchida após revisão explícita."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as case_file:
        reader = csv.DictReader(case_file)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Colunas ausentes nos casos manuais: {sorted(missing)}")

        cases: list[dict[str, str]] = []
        seen_nicknames: set[str] = set()
        for line_number, raw_row in enumerate(reader, start=2):
            row = {key: value or "" for key, value in raw_row.items()}
            nickname = row["nickname"].strip()
            expected_label = row["expected_label"].strip()
            review_status = row["review_status"].strip()
            if not nickname:
                raise ValueError(f"Apelido vazio na linha {line_number}.")
            normalized = normalize_nickname(nickname)
            if normalized in seen_nicknames:
                raise ValueError(
                    f"Apelido duplicado após normalização na linha {line_number}."
                )
            seen_nicknames.add(normalized)
            if review_status == "pendente" and expected_label:
                raise ValueError(
                    f"Rótulo de referência pendente não pode ser preenchido "
                    f"(linha {line_number})."
                )
            if review_status == "revisado" and expected_label not in VALID_LABELS:
                raise ValueError(
                    f"Rótulo válido obrigatório após revisão (linha {line_number})."
                )
            if review_status not in {"pendente", "revisado"}:
                raise ValueError(f"Estado de revisão inválido na linha {line_number}.")
            if review_status == "revisado" and not row["notes"].strip():
                raise ValueError(
                    f"Justificativa obrigatória para caso revisado "
                    f"(linha {line_number})."
                )
            cases.append(
                {
                    "nickname": nickname,
                    "expected_label": expected_label if review_status == "revisado" else "",
                    "review_status": review_status,
                    "notes": row["notes"].strip(),
                    "root_group": row["root_group"].strip(),
                }
            )
    return cases


def _base_presence(
    nickname: str,
    base_rows: Sequence[Mapping[str, str]],
    training_nicknames: set[str],
    holdout_nicknames: set[str],
) -> tuple[str, str, str]:
    normalized = normalize_nickname(nickname)
    matches = [
        row
        for row in base_rows
        if normalize_nickname(row.get("nickname", "")) == normalized
    ]
    if not matches:
        return "não encontrado na base autoral", "", ""

    matched_names = "; ".join(row.get("nickname", "") for row in matches)
    matched_groups = "; ".join(
        sorted({row.get("root_group", "") for row in matches if row.get("root_group")})
    )
    eligible_matches = [
        row
        for row in matches
        if row.get("review_status") == "revisado"
        and row.get("label") in {"permitido", "inadequado"}
    ]
    if any(
        normalize_nickname(row["nickname"]) in training_nicknames
        for row in eligible_matches
    ):
        return "já presente no treino", matched_names, matched_groups
    if any(
        normalize_nickname(row["nickname"]) in holdout_nicknames
        for row in eligible_matches
    ):
        return "já presente no holdout", matched_names, matched_groups
    return "presente, mas excluído da avaliação binária", matched_names, matched_groups


def diagnose_manual_cases(
    cases: Sequence[Mapping[str, str]],
    model: Any,
    *,
    base_rows: Sequence[Mapping[str, str]],
    training_rows: Sequence[Mapping[str, str]],
    holdout_rows: Sequence[Mapping[str, str]],
    candidate_terms: list[LexicalCandidate] | None = None,
    candidate_reviews: Mapping[int, CandidateReview] | None = None,
) -> list[dict[str, str]]:
    """Compara previsão, regra lexical, decisão final e referência revisada."""
    training_nicknames = {
        normalize_nickname(row["nickname"]) for row in training_rows
    }
    holdout_nicknames = {
        normalize_nickname(row["nickname"]) for row in holdout_rows
    }
    report_rows: list[dict[str, str]] = []

    for case in cases:
        nickname = case["nickname"]
        result = predict_nickname(
            nickname,
            model=model,
            candidate_terms=candidate_terms,
            candidate_reviews=candidate_reviews,
        )
        probabilities = result["probabilities"] or {}
        predicted = result["predicted_class"] or ""
        final_result = (
            "INADEQUADO"
            if result["automatic_block"]
            else (
                "REVISÃO NECESSÁRIA"
                if result["uncertain"]
                else RESULT_MESSAGES.get(predicted, "REVISÃO NECESSÁRIA")
            )
        )
        reasons = result["uncertainty_reason"]
        if isinstance(reasons, str):
            reasons = [reasons]
        base_presence, base_match, base_group = _base_presence(
            nickname, base_rows, training_nicknames, holdout_nicknames
        )
        report_rows.append(
            {
                "nickname": nickname,
                "model_prediction": predicted,
                "probability_permitido": (
                    f"{probabilities['permitido']:.6f}"
                    if "permitido" in probabilities
                    else ""
                ),
                "probability_inadequado": (
                    f"{probabilities['inadequado']:.6f}"
                    if "inadequado" in probabilities
                    else ""
                ),
                "lexical_status": str(result["lexical_result"]["status"]),
                "lexical_matches": json.dumps(
                    result["lexical_result"]["matches"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "final_result": final_result,
                "review_reason": "; ".join(
                    REASON_MESSAGES.get(reason, reason) for reason in reasons
                ),
                "base_presence": base_presence,
                "base_match_nickname": base_match,
                "base_root_group": base_group,
                "expected_label": (
                    case["expected_label"]
                    if case.get("review_status") == "revisado"
                    else ""
                ),
                "review_status": case["review_status"],
                "notes": case.get("notes", ""),
            }
        )
    return report_rows


def write_diagnostic_report(
    rows: Sequence[Mapping[str, str]],
    path: str | Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Grava o relatório tabular sem alterar a base nem os rótulos de referência."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gera um diagnóstico separado dos casos manuais pendentes."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    cases = load_manual_cases(args.cases)
    model = load_saved_model(args.model)
    _, base_rows = load_dataset(DEFAULT_DATASET)
    training_rows = eligible_rows(DEFAULT_DATASET)
    training_split, holdout_split = split_records_by_root_group(training_rows)
    report_rows = diagnose_manual_cases(
        cases,
        model,
        base_rows=base_rows,
        training_rows=training_split,
        holdout_rows=holdout_split,
    )
    destination = write_diagnostic_report(report_rows, args.output)
    pending_count = sum(row["review_status"] == "pendente" for row in cases)
    print(
        f"Diagnóstico de {len(report_rows)} casos salvo em {destination}. "
        f"Casos pendentes sem rótulo de referência: {pending_count}. "
        "Nenhum treinamento foi executado."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
