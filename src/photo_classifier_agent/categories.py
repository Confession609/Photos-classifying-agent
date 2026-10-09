"""Stable category definitions and zero-shot prompt templates."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CategoryDefinition:
    name: str
    description: str
    prompts: tuple[str, ...]


CATEGORIES: tuple[CategoryDefinition, ...] = (
    CategoryDefinition(
        name="人像",
        description="以人物为主要视觉主体，包括人物剪影。人物与星空同框时，人物主体优先。",
        prompts=(
            "a portrait photograph where a person is the main subject",
            "a photograph focused on one or more people",
            "a human silhouette as the main subject of a photograph",
        ),
    ),
    CategoryDefinition(
        name="风光摄影",
        description="以自然景物或自然风景为主要主体，但不包括以夜空为主要内容的照片。",
        prompts=(
            "a nature photograph of a landscape or natural scene without a starry night sky",
            "a photograph focused on natural scenery such as mountains, forests, rivers, or flowers",
            "a natural object or daylight landscape as the main subject",
        ),
    ),
    CategoryDefinition(
        name="星空",
        description="夜空或星空景象是主要拍摄内容，且没有更优先的人物主体。",
        prompts=(
            "a night sky or astrophotography image where stars are the main subject",
            "a photograph of the Milky Way or a star-filled sky",
            "a landscape dominated by the night sky and stars",
        ),
    ),
    CategoryDefinition(
        name="静物摄影",
        description="以人为制作的艺术品、物品或其他人工对象为主要主体，而非自然景物。",
        prompts=(
            "a still life photograph focused on a man-made object or artwork",
            "a photograph of an artificial object, sculpture, craft, or product as the main subject",
            "an arranged still life composition of human-made items",
        ),
    ),
)

CATEGORY_NAMES: tuple[str, ...] = tuple(category.name for category in CATEGORIES)
CATEGORY_BY_NAME = {category.name: category for category in CATEGORIES}
