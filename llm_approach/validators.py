"""Code-side validation of the model's slots. Nothing reaches the rooms API unchecked."""

from __future__ import annotations

import difflib
import re
from typing import Any

from llm_approach import reuse
from llm_approach.schema import Slots

BHK_MIN, BHK_MAX = 1, 5
PRICE_MIN, PRICE_MAX = 500, 500_000
FUZZY_CUTOFF = 0.85
MAX_CITY_WORDS = 3

# Words that mean "no real city here". "new" is deliberately absent (New Delhi).
_FILLER_TOKENS = frozenset(
    """pls plz please urgent urgently asap now immediately a an the different
    another any some same other area areas place places location locality city
    somewhere anywhere nearby near here there i dont do not know idk
    unknown none nothing nope yes no ok okay room rooms flat flats cheap
    cheaper good best something anything me my it this that one in of to for
    and with hi hello thanks thank""".split()
)

_CITY_RE = re.compile(r"^[a-z]+(?:[ .'-][a-z]+)*$")


def _city_candidates() -> dict[str, str]:
    """lowercase name or alias -> canonical lowercase name."""
    table = {name: name for name in reuse.get_known_cities()}
    for alias, canonical in reuse.get_city_aliases().items():
        table[alias.lower()] = canonical.lower()
    return table


def validate_city(raw: Any) -> str | None:
    """Return a title-cased city, or None when ``raw`` is not a usable city."""
    if not isinstance(raw, str):
        return None
    text = re.sub(r"\s+", " ", raw.lower()).strip(" .,;:!?\"'")
    if not text:
        return None

    table = _city_candidates()
    if text in table:
        return table[text].title()

    if len(text) >= 3:
        for match in difflib.get_close_matches(text, table, n=3, cutoff=FUZZY_CUTOFF):
            if match[0] == text[0]:
                return table[match].title()

    if not _CITY_RE.match(text) or len(text) < 3:
        return None
    if len(text.split()) > MAX_CITY_WORDS:
        return None
    if any(token in _FILLER_TOKENS for token in re.split(r"[ .'-]+", text) if token):
        return None
    return text.title()


def validate_bhk(raw: Any) -> int | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, str):
        match = re.fullmatch(r"\s*(\d+)\s*(?:bhk|rk)?\s*", raw.lower())
        if not match:
            return None
        raw = int(match.group(1))
    if isinstance(raw, float) and raw.is_integer():
        raw = int(raw)
    if isinstance(raw, int) and BHK_MIN <= raw <= BHK_MAX:
        return raw
    return None


_PRICE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(k|thousand|lakhs?|lacs?|l)?", re.IGNORECASE
)


def parse_price(raw: Any) -> int | None:
    """Rupee amount as an int (``"15k"`` -> 15000), or None if not a price."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, str):
        text = raw.lower().replace(",", "").replace("₹", "").strip()
        text = re.sub(r"^(rs\.?|inr)\s*", "", text)
        match = _PRICE_RE.fullmatch(text)
        if not match:
            return None
        unit = (match.group(2) or "").lower()
        factor = 1 if not unit else 1_000 if unit in ("k", "thousand") else 100_000
        value = float(match.group(1)) * factor
    else:
        return None
    if value != value or abs(value) == float("inf"):
        return None
    return int(round(value))


def _valid_price(raw: Any) -> int | None:
    value = parse_price(raw)
    return value if value is not None and PRICE_MIN <= value <= PRICE_MAX else None


# The API's amenity vocabulary (observed in the rooms payload): mostly
# lowercase-hyphenated, with a few capitalised entries.
AMENITY_VOCAB = {
    name.lower(): name
    for name in (
        "WiFi Gym Security hot-water bike-storage parking fan balcony window "
        "power-backup furnished semi-furnished unfurnished study-table lighting "
        "water-purifier gas-stove attached-bathroom shared-bathroom wardrobe fridge"
    ).split()
}
_AMENITY_ALIASES = {
    "wi-fi": "wifi", "wi fi": "wifi", "internet": "wifi", "broadband": "wifi",
    "geyser": "hot-water", "hot water": "hot-water", "heater": "hot-water",
    "bike parking": "bike-storage", "two wheeler parking": "bike-storage",
    "bike": "bike-storage", "bike storage": "bike-storage",
    "car parking": "parking", "inverter": "power-backup", "backup": "power-backup",
    "power backup": "power-backup", "generator": "power-backup",
    "ro": "water-purifier", "purifier": "water-purifier",
    "water purifier": "water-purifier", "study table": "study-table",
    "table": "study-table", "gas stove": "gas-stove", "stove": "gas-stove",
    "attached bathroom": "attached-bathroom", "attached bath": "attached-bathroom",
    "attached washroom": "attached-bathroom", "shared bathroom": "shared-bathroom",
    "semi furnished": "semi-furnished", "refrigerator": "fridge",
    "cupboard": "wardrobe", "almirah": "wardrobe", "cctv": "security",
    "gymnasium": "gym", "light": "lighting",
}


def validate_amenity(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    text = re.sub(r"[\s_]+", " ", raw.lower()).strip()
    text = _AMENITY_ALIASES.get(text, text)
    return AMENITY_VOCAB.get(text) or AMENITY_VOCAB.get(text.replace(" ", "-"))


_IDEAL_FOR = {
    "student": "Students", "students": "Students",
    "professional": "Working Professionals", "professionals": "Working Professionals",
    "working professional": "Working Professionals",
    "working professionals": "Working Professionals",
    "family": "Family", "families": "Family",
}


def validate_ideal_for(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    return _IDEAL_FOR.get(re.sub(r"\s+", " ", raw.lower()).strip())


def validate_room_reference(
    ref: Any, rooms_count: int, has_selected: bool
) -> int | str | None:
    """A 1-based index within the rooms on screen, or "selected" when one is."""
    if ref == "selected":
        return "selected" if has_selected else None
    if isinstance(ref, int) and not isinstance(ref, bool) and 1 <= ref <= rooms_count:
        return ref
    return None


def validate_slots(
    slots: Slots | dict | None,
    *,
    rooms_count: int = 0,
    has_selected: bool = False,
    room_reference: Any = None,
    compare_references: list | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Validate everything the model proposed.

    Returns ``(clean, rejected)``. ``clean`` has the six slot keys plus
    ``room_reference`` and ``compare_references``; a value that failed is None
    (or dropped from the compare list) and its name is listed in ``rejected``.
    A value the model left out is not counted as rejected. Reversed min/max
    prices are swapped, not rejected.
    """
    data = slots.model_dump() if isinstance(slots, Slots) else dict(slots or {})
    rejected: list[str] = []

    def check(name: str, raw: Any, validator) -> Any:
        if raw is None:
            return None
        value = validator(raw)
        if value is None:
            rejected.append(name)
        return value

    clean: dict[str, Any] = {
        "city": check("city", data.get("city"), validate_city),
        "bhk": check("bhk", data.get("bhk"), validate_bhk),
        "min_price": check("min_price", data.get("min_price"), _valid_price),
        "max_price": check("max_price", data.get("max_price"), _valid_price),
        "amenity": check("amenity", data.get("amenity"), validate_amenity),
        "ideal_for": check("ideal_for", data.get("ideal_for"), validate_ideal_for),
    }
    low, high = clean["min_price"], clean["max_price"]
    if low is not None and high is not None and low > high:
        clean["min_price"], clean["max_price"] = high, low

    clean["room_reference"] = check(
        "room_reference",
        room_reference,
        lambda r: validate_room_reference(r, rooms_count, has_selected),
    )
    refs = list(compare_references or [])
    good = [r for r in refs if validate_room_reference(r, rooms_count, False) is not None]
    if len(good) != len(refs):
        rejected.append("compare_references")
    clean["compare_references"] = good
    return clean, rejected
