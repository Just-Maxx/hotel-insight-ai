"""Контракты коммуникационного модуля. Утверждения отельера — не проверенная геометрия."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Feature = str  # допустимые id задаёт features/registry.yaml (проверка — в HotelierFact)
Source = Literal['hotelier_simulator', 'hotelier_human', 'hotel_website']


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, strict=True)


class PetPolicy(Contract):
    allowed: bool | None = None
    animals: list[Literal['dog', 'cat', 'other']] = Field(default_factory=list)
    max_weight_kg: float | None = Field(default=None, ge=0)
    weight_unlimited: bool | None = None
    fee_rub_per_day: float | None = Field(default=None, ge=0)
    needs_approval: bool | None = None

    @model_validator(mode='after')
    def consistent_weight(self):
        if self.weight_unlimited is True and self.max_weight_kg is not None:
            raise ValueError('Ограничение веса и отсутствие ограничения несовместимы')
        return self


class PolicyExtraction(Contract):
    policy: PetPolicy = Field(default_factory=PetPolicy)
    evidence: list[str] = Field(default_factory=list)
    conflict: bool = False
    needs_follow_up: bool = True
    follow_up: str | None = None

    @model_validator(mode='after')
    def require_evidence(self):
        if any(v is not None and v != [] for v in self.policy.model_dump().values()) and not self.evidence:
            raise ValueError('Факты требуют цитату')
        return self


class Claim(Contract):
    detail: str = Field(min_length=1, max_length=2000)
    quote: str = Field(min_length=1, max_length=4000)


class FactExtraction(Contract):
    status: Literal['reported', 'unknown', 'conflict']
    claims: list[Claim] = Field(default_factory=list)
    # 0..1: насколько утверждение говорит «характеристика присутствует» (для ранжирования); null = не оценить
    suitability: float | None = Field(default=None, ge=0, le=1)
    pet_policy: PolicyExtraction | None = None

    @model_validator(mode='after')
    def evidence_required(self):
        if self.status == 'reported' and not self.claims:
            raise ValueError('Утверждение требует цитату')
        if self.status != 'reported' and self.pet_policy is not None:
            raise ValueError('Спорная/неизвестная политика не должна содержать значения')
        return self


class Turn(Contract):
    question: str
    answer: str


class HotelierFact(Contract):
    feature: Feature
    status: Literal['reported', 'unknown', 'conflict', 'error']
    claims: list[Claim] = Field(default_factory=list)
    suitability: float | None = Field(default=None, ge=0, le=1)
    pet_policy: PetPolicy | None = None
    needs_follow_up: bool = False
    follow_up: str | None = None
    error: str | None = None

    @model_validator(mode='after')
    def known_feature(self):
        from core.registry import load_registry
        if self.feature not in load_registry():
            raise ValueError(f'Неизвестная характеристика: {self.feature}')
        return self


class HotelierSession(Contract):
    hotel_id: str = Field(min_length=1, max_length=200)
    source: Source
    source_url: str | None = None
    fetched_at: str
    turns: list[Turn]
    facts: list[HotelierFact]
    status: Literal['complete', 'needs_follow_up', 'error']
    verified: Literal[False] = False


class HotelierResult(Contract):
    """Совместимость с первым демо одного вопроса."""
    hotel_id: str
    feature: Literal['pets_allowed'] = 'pets_allowed'
    source: Source
    question: str
    answer: str
    extraction: PolicyExtraction
    fetched_at: str
