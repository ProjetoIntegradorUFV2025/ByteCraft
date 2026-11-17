"""API HTTP local para consultar o modelo experimental de apelidos."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .nickname_model import load_saved_model, predict_nickname


ModerationStatus = Literal["inadequado", "permitido", "revisao_necessaria"]

UNCERTAINTY_MESSAGES = {
    "probabilidade_indisponivel": "O modelo não forneceu uma estimativa de confiança.",
    "confianca_abaixo_do_limite": (
        "A confiança da previsão ficou abaixo do limite mínimo configurado."
    ),
    "correspondencia_lexical_requer_revisao": (
        "Uma correspondência lexical requer revisão humana."
    ),
    "discordancia_com_regra_lexical_revisada": (
        "A previsão divergiu de uma regra lexical revisada."
    ),
    "modelo_nao_disponivel": "O modelo de moderação não está disponível.",
}


class ModerationRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    nickname: Annotated[str, Field(min_length=1, max_length=50)]

    @field_validator("nickname")
    @classmethod
    def nickname_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("O apelido não pode ficar vazio.")
        return value.strip()


class ModerationResponse(BaseModel):
    status: ModerationStatus
    justificativa: str


def _moderation_response(prediction: dict[str, object]) -> ModerationResponse:
    if prediction.get("automatic_block") is True:
        return ModerationResponse(
            status="inadequado",
            justificativa=(
                "O apelido corresponde a um termo classificado como inadequado "
                "em um exemplo revisado."
            ),
        )

    if prediction.get("uncertain", True):
        reasons = prediction.get("uncertainty_reason", [])
        if isinstance(reasons, str):
            reasons = [reasons]
        messages = [
            UNCERTAINTY_MESSAGES.get(str(reason), "A previsão exige avaliação adicional.")
            for reason in reasons
        ]
        explanation = " ".join(dict.fromkeys(messages)) or (
            "A previsão exige avaliação adicional."
        )
        return ModerationResponse(
            status="revisao_necessaria",
            justificativa=explanation,
        )

    predicted_class = prediction.get("predicted_class")
    if predicted_class == "inadequado":
        return ModerationResponse(
            status="inadequado",
            justificativa=(
                "O modelo classificou o apelido como inadequado, sem sinalização "
                "de incerteza."
            ),
        )
    if predicted_class == "permitido":
        return ModerationResponse(
            status="permitido",
            justificativa=(
                "O modelo classificou o apelido como permitido, sem sinalização "
                "de incerteza."
            ),
        )
    return ModerationResponse(
        status="revisao_necessaria",
        justificativa="A previsão não corresponde a uma classe reconhecida.",
    )


def create_app(
    model_loader: Callable[[], object] = load_saved_model,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.nickname_model = model_loader()
        yield

    app = FastAPI(
        title="ByteCraft Nickname Moderation",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    @app.post("/moderate-nickname", response_model=ModerationResponse)
    def moderate_nickname(request: ModerationRequest) -> ModerationResponse:
        prediction = predict_nickname(
            request.nickname,
            model=app.state.nickname_model,
        )
        return _moderation_response(prediction)

    return app


app = create_app()
