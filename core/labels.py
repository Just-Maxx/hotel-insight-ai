"""Русские названия для интерфейса: пользователь не должен видеть id характеристик и служебные слова."""
from __future__ import annotations

from core.registry import load_registry

DEFAULT_WALK_MINUTES = 15   # «пешая доступность» без уточнения = не дальше 15 минут пешком
CLOSE_WALK_MINUTES = 5      # «у самого моря», «первая линия»

# Короткие названия встроенных характеристик (для карточек и сравнения).
FEATURE_LABEL = {
    "sunset_view": "Вид на закат",
    "sea_view": "Вид на море",
    "quiet": "Тишина",
    "pets_allowed": "Можно с питомцем",
    "walk_access": "Дорога пешком",
    "night_amenities": "Работает ночью",
    "kids_infrastructure": "Для детей",
}
WALK_TARGET_LABEL = {"beach": "Пляж", "bus_stop": "Остановка", "station": "Ж/д станция"}
WALK_TARGET_GENITIVE = {"beach": "до пляжа", "bus_stop": "до остановки", "station": "до станции"}

IMPORTANCE_LABEL = {"must": "важно", "nice": "желательно"}


def feature_label(feature: str) -> str:
    if feature in FEATURE_LABEL:
        return FEATURE_LABEL[feature]
    return load_registry().get(feature, {}).get("title", feature)


def requirement_label(feature: str, target: str | None = None, max_minutes: int | None = None) -> str:
    """«Пляж пешком, до 15 мин» для пожелания про дорогу, иначе — название характеристики."""
    if feature == "walk_access" and target:
        base = f"{WALK_TARGET_LABEL[target]} пешком"
        return f"{base}, до {max_minutes} мин" if max_minutes else base
    return feature_label(feature)
