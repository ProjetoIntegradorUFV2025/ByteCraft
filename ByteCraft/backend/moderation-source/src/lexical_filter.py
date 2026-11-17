"""Filtro lexical consultivo sobre candidatos locais do LDNOOBW."""

from __future__ import annotations

import csv
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .nickname_normalization import analyze_nickname, normalize_nickname


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = PROJECT_ROOT / "data" / "raw" / "ldnoobw" / "pt"
DEFAULT_REVIEW_LEDGER = (
    PROJECT_ROOT / "data" / "curated" / "ldnoobw_candidate_reviews.csv"
)
CLASSIFICATIONS = {
    "inadequado_claro",
    "ambiguo",
    "neutro",
    "pendente_revisao",
}


@dataclass(frozen=True)
class LexicalCandidate:
    """Uma entrada bruta usada somente como candidato de comparação."""

    source_line: int
    term: str
    normalized_term: str


@dataclass(frozen=True)
class CandidateReview:
    """Decisão humana separada do arquivo original de candidatos."""

    classification: str
    reason: str


@dataclass(frozen=True)
class LexicalMatch:
    """Correspondência com explicação e referência à linha da fonte."""

    source_line: int
    classification: str
    reason: str
    rule: str


def load_candidates(path: str | Path = DEFAULT_CANDIDATES) -> list[LexicalCandidate]:
    """Lê as entradas sem alterar a fonte; todas começam pendentes de revisão."""
    candidates: list[LexicalCandidate] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as source:
        for line_number, raw_line in enumerate(source, start=1):
            term = raw_line.rstrip("\r\n")
            if term.strip():
                candidates.append(
                    LexicalCandidate(
                        source_line=line_number,
                        term=term,
                        normalized_term=normalize_nickname(term),
                    )
                )
    return candidates


def load_candidate_reviews(
    path: str | Path = DEFAULT_REVIEW_LEDGER,
) -> dict[int, CandidateReview]:
    """Carrega decisões humanas; IDs são números de linha do arquivo-fonte."""
    reviews: dict[int, CandidateReview] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as ledger:
        reader = csv.DictReader(ledger)
        required = {"source_line", "classification", "reason"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Colunas ausentes no registro de revisão: {sorted(missing)}")
        for row in reader:
            try:
                source_line = int(row["source_line"])
            except (TypeError, ValueError) as error:
                raise ValueError("source_line deve ser um número inteiro.") from error
            classification = row["classification"]
            reason = row["reason"].strip()
            if source_line < 1:
                raise ValueError("source_line deve ser positivo.")
            if source_line in reviews:
                raise ValueError(f"source_line duplicado no registro: {source_line}.")
            if classification not in CLASSIFICATIONS:
                raise ValueError(
                    f"classification inválida para source_line {source_line}."
                )
            if not reason:
                raise ValueError(
                    f"reason vazia para source_line {source_line}."
                )
            reviews[source_line] = CandidateReview(classification, reason)
    return reviews


def _phrase_occurs(candidate_tokens: tuple[str, ...], nick_tokens: tuple[str, ...]) -> bool:
    if not candidate_tokens or len(candidate_tokens) > len(nick_tokens):
        return False
    width = len(candidate_tokens)
    return any(
        nick_tokens[index : index + width] == candidate_tokens
        for index in range(len(nick_tokens) - width + 1)
    )


def _match_rule(candidate: LexicalCandidate, nickname: str) -> str | None:
    forms = analyze_nickname(nickname)
    candidate_forms = analyze_nickname(candidate.normalized_term)
    candidate_tokens = candidate_forms.tokens
    if not candidate_tokens:
        return None

    if _phrase_occurs(candidate_tokens, forms.tokens):
        return "limite_de_palavra_exato"
    leet_was_mapped = forms.leet_mapped != forms.accentless
    nickname_contains_letter = any(character.isalpha() for character in forms.accentless)
    if (
        leet_was_mapped
        and nickname_contains_letter
        and _phrase_occurs(candidate_tokens, forms.leet_tokens)
    ):
        return "substituicao_leetspeak_com_limite_de_palavra"

    compact_candidate = "".join(candidate_tokens)
    if (
        len(candidate_tokens) == 1
        and compact_candidate in forms.separator_compacts
    ):
        return "separadores_inseridos_entre_letras"
    if (
        len(candidate_tokens) == 1
        and compact_candidate in forms.repeated_collapsed_tokens
    ):
        return "letras_repetidas_requerem_revisao"
    return None


def check_nickname(
    nickname: str,
    candidates: list[LexicalCandidate] | None = None,
    reviews: Mapping[int, CandidateReview] | None = None,
) -> dict[str, object]:
    """Retorna parecer consultivo; candidatos sem revisão nunca são bloqueio."""
    loaded_candidates = candidates if candidates is not None else load_candidates()
    human_reviews = reviews or {}
    candidate_lines = {candidate.source_line for candidate in loaded_candidates}
    stale_reviews = set(human_reviews).difference(candidate_lines)
    if stale_reviews:
        raise ValueError(
            "Revisões apontam para linhas inexistentes na fonte: "
            f"{sorted(stale_reviews)}."
        )
    matches: list[LexicalMatch] = []
    for candidate in loaded_candidates:
        rule = _match_rule(candidate, nickname)
        if rule is None:
            continue
        review = human_reviews.get(candidate.source_line)
        if rule == "letras_repetidas_requerem_revisao":
            classification = "pendente_revisao"
            reason = (
                "A correspondência depende de reduzir letras repetidas; "
                "requer revisão humana."
            )
        elif review is None:
            classification = "pendente_revisao"
            reason = "Correspondência com candidato ainda não revisado por pessoa."
        else:
            if review.classification not in CLASSIFICATIONS or not review.reason.strip():
                raise ValueError(
                    f"Revisão inválida para source_line {candidate.source_line}."
                )
            classification = review.classification
            reason = review.reason
        matches.append(
            LexicalMatch(
                source_line=candidate.source_line,
                classification=classification,
                reason=reason,
                rule=rule,
            )
        )

    priority = {
        "inadequado_claro": 0,
        "ambiguo": 1,
        "pendente_revisao": 2,
        "neutro": 3,
    }
    if not matches:
        status = "neutro"
        reason = "Nenhum candidato lexical correspondeu com as regras atuais."
    else:
        status = min(matches, key=lambda match: priority[match.classification]).classification
        reason = next(match.reason for match in matches if match.classification == status)

    return {
        "nickname_original": nickname,
        "status": status,
        "reason": reason,
        "matches": [
            {
                "source_line": match.source_line,
                "classification": match.classification,
                "reason": match.reason,
                "rule": match.rule,
            }
            for match in matches
        ],
        "automatic_block": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Consulta lexical consultiva; nunca bloqueia automaticamente."
    )
    parser.add_argument("nickname", help="Apelido fictício ou a ser avaliado.")
    args = parser.parse_args()
    result = check_nickname(
        args.nickname,
        candidates=load_candidates(),
        reviews=load_candidate_reviews(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
