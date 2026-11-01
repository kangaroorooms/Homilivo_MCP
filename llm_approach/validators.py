"""Code-side validation of the model's slots. Nothing reaches the rooms API unchecked."""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from typing import Any

from llm_approach import extra_security, reply_safety, reuse
from llm_approach.sanitize import neutralize
from llm_approach.schema import FACT_KEYS, Slots

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


def _is_known_city(raw: Any) -> bool:
    return isinstance(raw, str) and re.sub(r"\s+", " ", raw.lower()).strip(" .,;:!?\"'") in _city_candidates()


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


MAX_AREA_WORDS = 4
AREA_MIN_CHARS, AREA_MAX_CHARS = 2, 40
_AREA_RE = re.compile(r"^[a-z0-9]+(?:[ .-]+[a-z0-9]+)*\.?$")
_AREA_FILLER = frozenset(
    """different any anywhere other another somewhere something anything
    pls plz please urgent urgently asap idk unknown none nothing nope yes no
    ok okay""".split()
)
_AREA_FILLER_PHRASES = frozenset(
    {"area", "areas", "place", "location", "locality", "same", "same area", "nearby", "near me",
     "i dont know", "i do not know", "dont know", "do not know"}
)


def validate_area(raw: Any) -> str | None:
    """A title-cased locality inside a city, or None.

    Filler ("different area", "any", "pls"), a known city or city alias, and
    anything that is not 2 to 40 characters of letters, digits, spaces,
    hyphens and dots (at most 4 words) is not an area."""
    if not isinstance(raw, str):
        return None
    text = re.sub(r"\s+", " ", raw.lower()).strip(" ,;:!?\"'")
    if not (AREA_MIN_CHARS <= len(text) <= AREA_MAX_CHARS) or not _AREA_RE.match(text):
        return None
    words = re.split(r"[ .-]+", text.strip(". "))
    words = [w for w in words if w]
    if not words or len(text.split()) > MAX_AREA_WORDS:
        return None
    if text in _AREA_FILLER_PHRASES or any(w in _AREA_FILLER for w in words):
        return None
    if text in _city_candidates():
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

    Returns ``(clean, rejected)``. ``clean`` has the seven slot keys (area included) plus
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
    raw_area = data.get("area")
    clean["area"] = validate_area(raw_area)
    if raw_area is not None and clean["area"] is None:
        # A city put in the area slot is a city, not a bad value.
        as_city = validate_city(raw_area) if _is_known_city(raw_area) else None
        if as_city and clean["city"] is None:
            clean["city"] = as_city
        elif not as_city:
            rejected.append("area")
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


# ---- user facts (session memory) -----------------------------------------------------

NAME_MAX_CHARS, NAME_MAX_WORDS = 40, 4
AGE_MIN, AGE_MAX = 16, 100
OCCUPATION_MIN_CHARS, OCCUPATION_MAX_CHARS, OCCUPATION_MAX_WORDS = 2, 40, 5
_NAME_PUNCT = " .'-"
_OCCUPATION_PUNCT = " .'-&/"
_AGE_RE = re.compile(r"(\d{1,3})(?:\s*(?:years?|yrs?|yo|y/o)(?:\s*old)?)?", re.IGNORECASE)
_GENDERS = {
    "male": "male", "man": "male", "boy": "male", "m": "male",
    "female": "female", "woman": "female", "girl": "female", "f": "female",
    "non-binary": "non-binary", "nonbinary": "non-binary", "non binary": "non-binary",
    "other": "other",
}
_NUMBER_WORDS = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|lakh|lac)\b",
    re.IGNORECASE,
)


def _plain_text(raw: Any, allowed_punct: str, min_chars: int, max_chars: int, max_words: int) -> str | None:
    """Letters (any script, with combining marks) plus ``allowed_punct``; None otherwise.

    Rejects, rather than cleans, anything with invisible/format/control characters (zero-width
    joiners, tag characters), digits, '@', angle brackets, colons or brackets, and any text the
    injection filter would block (role prefixes, "ignore previous instructions", fence tags)."""
    if not isinstance(raw, str):
        return None
    text = re.sub(r"\s+", " ", raw).strip()
    if not (min_chars <= len(text) <= max_chars) or len(text.split()) > max_words:
        return None
    for char in text:
        category = unicodedata.category(char)
        if not (char.isalpha() or category.startswith("M") or char in allowed_punct):
            return None
    if not text[0].isalpha() or extra_security.is_extra_blocked(text)[0]:
        return None
    return text


def validate_name(raw: Any) -> str | None:
    text = _plain_text(raw, _NAME_PUNCT, 1, NAME_MAX_CHARS, NAME_MAX_WORDS)
    if text is None:
        return None
    return text.title() if text.islower() or text.isupper() else text


def validate_age(raw: Any) -> int | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, float) and raw.is_integer():
        raw = int(raw)
    if isinstance(raw, str):
        match = _AGE_RE.fullmatch(raw.strip())
        if not match:
            return None
        raw = int(match.group(1))
    return raw if isinstance(raw, int) and AGE_MIN <= raw <= AGE_MAX else None


def validate_occupation(raw: Any) -> str | None:
    text = _plain_text(raw, _OCCUPATION_PUNCT, OCCUPATION_MIN_CHARS, OCCUPATION_MAX_CHARS, OCCUPATION_MAX_WORDS)
    return text.lower() if text else None


def validate_gender(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    return _GENDERS.get(re.sub(r"\s+", " ", raw.lower()).strip())


_FACT_VALIDATORS = {
    "name": validate_name,
    "age": validate_age,
    "occupation": validate_occupation,
    "gender": validate_gender,
    "preferred_city": validate_city,
    "budget": _valid_price,
}
assert set(_FACT_VALIDATORS) == set(FACT_KEYS)


_WORD_VALUES = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()) if i > 1}
_NUMBER_WORD_RE = re.compile(
    r"\b(?:(" + "|".join(_TENS) + r")(?:[\s-]+(" + "|".join(w for w in _WORD_VALUES if 0 < _WORD_VALUES[w] < 10) + r"))?"
    r"|(" + "|".join(w for w in _WORD_VALUES if _WORD_VALUES[w] >= 0) + r"))\b"
)
_ME = r"(?:i\s*am|i'?m|im|me|main|mai|mein|myself)"
_AGE_CUES = (
    r"\b{me}\s+(?:about\s+|around\s+|almost\s+)?(?:a\s+)?{n}\b",
    r"\bage\s*(?:is|:|=|of)?\s*{n}\b",
    r"\bmy\s+age\s+(?:is\s+)?{n}\b",
    r"\b(?:aged|age)\s+{n}\b",
    r"\b{n}\s*(?:years?|yrs?|yo|y/o|saal|sal|varsh)\b",
    r"\b(?:meri|meri\s+umar|meri\s+umra|umar|umra)\s*(?:age\s*)?(?:is\s+)?{n}\b",
    r"\b{n}\s*(?:ka|ki)\s+hu+n?\b",
    r"\b{me}\s+[^\d.;!?]{{1,40}}?(?:,|\s+and|\s+aur|\s*&)\s*{n}\b",  # "I'm Faizal, 23, need a 2 BHK", "I'm Zed and 40"
)
_BUDGET_CUE = re.compile(
    r"\b(?:budget|under|below|upto|up\s+to|max(?:imum)?|within|around|about|afford|rs\.?|inr|rupees?|rent|price|tak|hazaar)\b|₹"
)
_NAME_CUE = re.compile(
    r"\b(?:name|naam|i\s*am|i'?m|im|call\s+me|this\s+is|myself|my\s+self|main|mai|it'?s|its|actually|correction|sorry|rather|instead)\b"
)
_OCCUPATION_CUE = re.compile(
    r"\b(?:i\s*am|i'?m|im|main|mai|work|works|working|occupation|profession|job|kaam|naukri|study|studying|student|"
    r"employee|engineer|doctor|teacher)\b")
_CITY_CUE = re.compile(
    r"\b(?:prefer|live|living|stay|staying|based|from|belong|moving|shifting|relocat\w*|looking|need|want|search\w*|find|"
    r"room|flat|pg|in|at|mein|me|ke)\b")
_GENDER_CUE = re.compile(r"\b(?:i\s*am|i'?m|im|gender|myself|main|mai|hu|hoon|am)\b")


def _numberize(flat: str) -> str:
    """Spoken numbers as digits ("twenty two" -> "22") so one pattern reads both forms."""
    def repl(m: re.Match) -> str:
        if m.group(1):
            return str(_TENS[m.group(1)] + (_WORD_VALUES[m.group(2)] if m.group(2) else 0))
        return str(_WORD_VALUES[m.group(3)])

    return _NUMBER_WORD_RE.sub(repl, flat)


def _age_stated(value: int, flat: str) -> bool:
    """The message states this age: "I am 22", "age 22", "22 years", "meri age 22 hai", "I'm twenty two"."""
    text = _numberize(flat)
    return any(re.search(p.format(me=_ME, n=rf"(?<![\d.]){value}(?![\d.])"), text) for p in _AGE_CUES)


def _budget_stated(value: int, flat: str) -> bool:
    """The message holds a price that equals this budget, with a budget cue ("budget 15k", "under 15000", "rs 15,000")."""
    if not _BUDGET_CUE.search(flat):
        return False
    text = _numberize(flat).replace(",", "")
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(k|thousand|hazaar|lakhs?|lacs?|l)?\b", text):
        unit = (m.group(2) or "").lower()
        factor = 1 if not unit else 1_000 if unit in ("k", "thousand", "hazaar") else 100_000
        if int(round(float(m.group(1)) * factor)) == value:
            return True
    return False


def _grounded(key: str, raw: Any, clean: Any, message: str) -> bool:
    """The value must come from the user's message, in a context that states it, not from the model's imagination."""
    flat = re.sub(r"\s+", " ", message.lower())
    if key == "age":
        return isinstance(clean, int) and _age_stated(clean, flat)
    if key == "budget":
        return isinstance(clean, int) and _budget_stated(clean, flat)
    if key == "gender":  # "I'm a woman" grounds "female"; one-letter synonyms never count
        toks = re.findall(r"[a-z]+", flat)  # single words and neighbouring pairs ("non binary")
        words = set(toks) | {f"{a} {b}" for a, b in zip(toks, toks[1:])} | {f"{a}-{b}" for a, b in zip(toks, toks[1:])}
        return _GENDER_CUE.search(flat) is not None and any(len(word) > 1 and _GENDERS.get(word) == clean for word in words)
    candidates = {re.sub(r"\s+", " ", str(v).lower()).strip() for v in (raw, clean) if isinstance(v, str)}
    if not any(c and c in flat for c in candidates):
        return False
    cue = {"name": _NAME_CUE, "occupation": _OCCUPATION_CUE, "preferred_city": _CITY_CUE}.get(key)
    return cue is None or cue.search(flat) is not None


def validate_user_facts(raw: Any, message: str | None = None) -> tuple[dict[str, Any], list[str]]:
    """Clean the facts the model extracted. Returns ``(clean, rejected)`` (key names only).

    Unknown keys are dropped silently. Values that fail their validator are rejected, and so is
    any value containing an e-mail address or a phone-number-like run of digits. With ``message``,
    a value that does not appear in the user's message is rejected as well."""
    clean: dict[str, Any] = {}
    rejected: list[str] = []
    if not isinstance(raw, dict):
        return clean, rejected
    for key in FACT_KEYS:
        value = raw.get(key)
        if value is None:
            continue
        if isinstance(value, str) and ("@" in value or len(re.findall(r"\d", value)) >= 7):
            rejected.append(key)  # e-mail or phone number: never stored
            continue
        checked = _FACT_VALIDATORS[key](value)
        if checked is None or (message is not None and not _grounded(key, value, checked, message)):
            rejected.append(key)
            continue
        clean[key] = checked
    return clean, rejected


# ---- conversation answers (CONVERSATION_QUESTION) --------------------------------------
#
# The answerer model returns {"found", "answer", "evidence_turns"}. Nothing it says reaches the
# user until this code has checked it against the transcript it was given.

NOT_IN_CHAT = "I don't have that in this chat."
ANSWER_MAX_CHARS = 400
ANSWER_MAX_SENTENCES = 3

_BOT_QUESTION = re.compile(
    r"\b(?:what|why|when|how|where)\s+(?:did|do|have|were|are)\s+you\b|\byou\s+(?:said|told|showed|mentioned|replied|suggested|recommended|answered)\b"
    r"|\byour\s+(?:reply|replies|answer|message|response|suggestion|recommendation)\b|\bwhy\s+(?:was|is)\s+(?:that|this|it)\b",
    re.IGNORECASE,
)
_USER_QUESTION = re.compile(r"\b(?:my|i|me|mine|i'm|i've|myself)\b", re.IGNORECASE)
_QUOTED = re.compile(r"[\"“”]([^\"“”]{3,})[\"“”]")
_PHONE_RUN = re.compile(r"\+?\d[\d\s().-]{5,}\d")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def parse_answer_json(raw: Any) -> dict[str, Any] | None:
    """The answerer's JSON as {"found": bool, "answer": str, "evidence_turns": [int]}, or None when
    it is not a JSON object of that shape (a code fence around it is tolerated)."""
    data = None
    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, str):
        text = raw.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        try:
            data = json.loads(fenced.group(1) if fenced else text)
        except (ValueError, TypeError):
            return None
    if not isinstance(data, dict) or not isinstance(data.get("found"), bool):
        return None
    answer = data.get("answer")
    if answer is None:
        answer = ""
    if not isinstance(answer, str):
        return None
    evidence = data.get("evidence_turns", [])
    if evidence is None:
        evidence = []
    if not isinstance(evidence, list) or any(isinstance(e, bool) or not isinstance(e, int) for e in evidence):
        return None
    return {"found": data["found"], "answer": answer, "evidence_turns": list(dict.fromkeys(evidence))}


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("’", "'")).strip()


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


_MARKUP = re.compile(r"[\[\]<>`]|//")
_NUM_TOKEN = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(k|thousand|lakhs?|lacs?)?(?![\w])", re.IGNORECASE)
_ROOM_COUNT_WORDS = re.compile(
    r"\b(" + "|".join(w for w in _WORD_VALUES if _WORD_VALUES[w] > 0) + r")\s+(?:more\s+|other\s+)?(?:rooms?|options?|listings?)\b",
    re.IGNORECASE,
)
# Things a reply may claim about a room: the amenity vocabulary and aliases, plus common ones outside it.
_CLAIM_WORDS = sorted(
    {k.replace("-", " ") for k in AMENITY_VOCAB} | set(_AMENITY_ALIASES) | {
        "pool", "swimming pool", "ac", "air conditioning", "air conditioner", "lift", "elevator", "tv", "television",
        "washing machine", "kitchen", "laundry", "meals", "food", "tiffin", "housekeeping", "garden", "terrace", "parking",
    },
    key=len,
    reverse=True,
)
_CLAIM_RE = re.compile(r"(?<![\w])(" + "|".join(re.escape(w) for w in _CLAIM_WORDS) + r")(?![\w])", re.IGNORECASE)


def _numbers_in(text: str) -> set[int]:
    """Whole numbers in a text, commas removed, "7k" read as 7000."""
    out: set[int] = set()
    for m in _NUM_TOKEN.finditer(text):
        unit = (m.group(2) or "").lower()
        factor = 1 if not unit else 1_000 if unit in ("k", "thousand") else 100_000
        try:
            out.add(int(round(float(m.group(1).replace(",", "")) * factor)))
        except ValueError:
            continue
    return out


def _norm_claim(text: str) -> str:
    return re.sub(r"[\s_-]+", " ", text.lower()).strip()


def _unsupported_claim(answer: str, known: str, known_numbers: set[int]) -> bool:
    """True when the answer states an amount, a number of rooms or an amenity that the record it was given does not hold."""
    allowed = _numbers_in(known) | known_numbers
    if any(n not in allowed for n in _numbers_in(answer)):
        return True
    if any(_WORD_VALUES[w.lower()] not in allowed for w in _ROOM_COUNT_WORDS.findall(answer)):
        return True
    seen = _norm_claim(known)
    return any(_norm_claim(c) not in seen for c in _CLAIM_RE.findall(answer))


def _cap_answer(text: str) -> str:
    """At most 3 sentences and 400 characters, cut at a sentence or word end."""
    parts = _SENTENCE_END.split(text.strip())
    text = " ".join(parts[:ANSWER_MAX_SENTENCES])
    if len(text) > ANSWER_MAX_CHARS:
        cut = text[:ANSWER_MAX_CHARS]
        end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        if end >= ANSWER_MAX_CHARS // 2:
            cut = cut[: end + 1]
        else:
            cut = cut[: cut.rfind(" ")] if " " in cut else cut
            cut = cut.rstrip(",;:- ") + "..."
        text = cut
    return text


def validate_conversation_answer(
    parsed: dict[str, Any] | None,
    transcript: list[dict[str, Any]],
    question: str,
    *,
    facts: dict[str, Any] | None = None,
    room_titles: list[str] | None = None,
    known_text: str | None = None,
    known_numbers: set[int] | None = None,
) -> str:
    """The reply to show: the model's answer if every check passes, else ``NOT_IN_CHAT``.

    ``transcript`` is the list of turns the model was shown ({turn, role, text}). Checks, in order:
    found is true; the answer is not empty; at least one cited turn and every cited turn exists;
    a question about the bot's reply cites a bot turn, any other question cites a user turn
    (a question about neither, such as a summary, cites any turn); quoted text, phone-like numbers
    and e-mail addresses in the answer appear in the cited turns (or in the stored facts / room
    titles); the answer is cleaned by ``sanitize.neutralize``, cut to 3 sentences / 400 characters
    and must not match the injection filter."""
    if parsed is None or not parsed["found"]:
        return NOT_IN_CHAT
    by_id = {t["turn"]: t for t in transcript}
    cited = parsed["evidence_turns"]
    if not cited or any(c not in by_id for c in cited):
        return NOT_IN_CHAT
    roles = {by_id[c]["role"] for c in cited}
    if _BOT_QUESTION.search(question):
        if "bot" not in roles:
            return NOT_IN_CHAT
    elif _USER_QUESTION.search(question) and "user" not in roles:
        return NOT_IN_CHAT

    answer = re.sub(r"\s+", " ", neutralize(parsed["answer"])).strip()
    if not answer:
        return NOT_IN_CHAT
    answer = _cap_answer(answer)
    if extra_security.is_extra_blocked(answer)[0]:
        return NOT_IN_CHAT

    source_text = " ".join(by_id[c]["text"] for c in cited)
    extra = list((facts or {}).values()) + list(room_titles or [])
    haystack = _flat(source_text + " " + " ".join(str(v) for v in extra))
    digit_haystack = _digits(source_text + " " + " ".join(str(v) for v in extra))
    for quoted in _QUOTED.findall(answer):
        if _flat(quoted) not in haystack:
            return NOT_IN_CHAT
    for run in _PHONE_RUN.findall(answer):
        digits = _digits(run)
        if len(digits) >= 7 and digits not in digit_haystack:
            return NOT_IN_CHAT
    for email in _EMAIL.findall(answer):
        if email.lower() not in haystack:
            return NOT_IN_CHAT
    # v2: the answer may only state what the model was given (the whole record it saw, and the question): no link or markup,
    # no amount, number of rooms or amenity that is not in it
    plain = answer.replace("[blocked-tag]", "")  # the neutraliser's own marker is not markup
    if reply_safety.find_unsafe(plain) or _MARKUP.search(plain):
        return NOT_IN_CHAT
    known = _flat(known_text) if known_text is not None else haystack + " " + _flat(question)
    if _unsupported_claim(answer, known, known_numbers or set()):
        return NOT_IN_CHAT
    return answer
