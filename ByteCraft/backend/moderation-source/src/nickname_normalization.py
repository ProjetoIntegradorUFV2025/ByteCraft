"""Transformações não destrutivas para comparar apelidos."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


_WORD_PATTERN = re.compile(r"[^\W_]+")
_LEET_TRANSLATION = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "@": "a",
        "$": "s",
    }
)


@dataclass(frozen=True)
class MatchForms:
    """Representações auxiliares, sem alterar o apelido recebido."""

    original: str
    unicode_normalized: str
    casefolded: str
    accentless: str
    leet_mapped: str
    tokens: tuple[str, ...]
    leet_tokens: tuple[str, ...]
    separator_compacts: tuple[str, ...]
    repeated_letter_runs: tuple[str, ...]
    repeated_collapsed_tokens: tuple[str, ...]
    has_inserted_separators: bool


def normalize_unicode(value: str) -> str:
    """Aplica normalização Unicode de compatibilidade NFKC."""
    return unicodedata.normalize("NFKC", value)


def casefold_text(value: str) -> str:
    """Compara texto sem distinção entre maiúsculas e minúsculas."""
    return value.casefold()


def remove_diacritics(value: str) -> str:
    """Remove marcas combinantes após decomposição; não translitera outras letras."""
    decomposed = unicodedata.normalize("NFD", value)
    without_marks = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    return unicodedata.normalize("NFC", without_marks)


def normalize_nickname(value: str) -> str:
    """Normaliza caixa, Unicode, acentos e espaços, preservando pontuação e números."""
    normalized = normalize_unicode(value)
    folded = casefold_text(normalized)
    accentless = remove_diacritics(folded)
    return " ".join(accentless.split())


def map_leetspeak(value: str) -> str:
    """Mapeia substituições comuns explicitamente; não remove caracteres."""
    return value.translate(_LEET_TRANSLATION)


def compact_separators(value: str) -> str:
    """Remove pontuação de uma sequência para detectar separadores inseridos."""
    return "".join(_WORD_PATTERN.findall(value))


def collapse_repeated_letters(value: str) -> str:
    """Colapsa repetições idênticas apenas para sugerir revisão no filtro."""
    return re.sub(r"([^\W\d_])\1+", r"\1", value)


def repeated_letter_runs(value: str) -> tuple[str, ...]:
    """Retorna sequências repetidas sem modificar o texto."""
    return tuple(
        match.group(0)
        for match in re.finditer(r"([^\W\d_])\1+", value)
    )


def _separator_compacts(value: str) -> tuple[str, ...]:
    """Detecta letras unidas por pontuação ou por espaços entre letras isoladas."""
    compacts: set[str] = set()
    for chunk in value.split():
        compact = compact_separators(chunk)
        if compact != chunk and compact:
            compacts.add(compact)

    tokens = _WORD_PATTERN.findall(value)
    if len(tokens) > 1 and all(len(token) == 1 for token in tokens):
        compacts.add("".join(tokens))
    return tuple(sorted(compacts))


def analyze_nickname(value: str) -> MatchForms:
    """Produz formas comparáveis e sinais; `original` sempre preserva a entrada."""
    unicode_normalized = normalize_unicode(value)
    folded = casefold_text(unicode_normalized)
    accentless = remove_diacritics(folded)
    leet_mapped = map_leetspeak(accentless)
    tokens = tuple(_WORD_PATTERN.findall(accentless))
    leet_tokens = tuple(_WORD_PATTERN.findall(leet_mapped))
    compacts = _separator_compacts(accentless)
    repeated = repeated_letter_runs(accentless)
    repeated_collapsed = tuple(
        collapse_repeated_letters(token) for token in tokens
    )
    has_inserted_separators = bool(compacts)

    return MatchForms(
        original=value,
        unicode_normalized=unicode_normalized,
        casefolded=folded,
        accentless=accentless,
        leet_mapped=leet_mapped,
        tokens=tokens,
        leet_tokens=leet_tokens,
        separator_compacts=compacts,
        repeated_letter_runs=repeated,
        repeated_collapsed_tokens=repeated_collapsed,
        has_inserted_separators=has_inserted_separators,
    )
