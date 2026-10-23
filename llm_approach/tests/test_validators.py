import pytest

from llm_approach.schema import Slots
from llm_approach.validators import (
    parse_price, validate_amenity, validate_bhk, validate_city, validate_ideal_for,
    validate_room_reference, validate_slots,
)


@pytest.mark.parametrize("raw,expected", [
    ("bangalor", "Bangalore"), ("Bangalore", "Bangalore"), ("  BANGALORE ", "Bangalore"),
    ("blr", "Bangalore"), ("bengaluru", "Bangalore"), ("mumbay", "Mumbai"),
    ("Udupi", "Udupi"), ("Vadodara", "Vadodara"), ("new delhi", "New Delhi"),
    ("navi mumbai", "Navi Mumbai"), ("pune.", "Pune"),
])
def test_city_accepted(raw, expected):
    assert validate_city(raw) == expected


@pytest.mark.parametrize("raw", [
    "Pls Urgent", "pls", "A Different Area", "I Dont Know", "somewhere", "anywhere nearby",
    "", "  ", "12345", "pune123", "a b c d", "ab", "<script>", None, 7,
])
def test_city_rejected(raw):
    assert validate_city(raw) is None


def test_fuzzy_requires_same_first_letter():
    assert validate_city("xangalore") != "Bangalore"


@pytest.mark.parametrize("raw,expected", [
    ("15k", 15000), ("15K", 15000), ("1.5 lakh", 150000), ("2 lac", 200000), ("Rs. 12,000", 12000),
    ("₹8000", 8000), (15000, 15000), (7500.0, 7500), ("15 thousand", 15000),
])
def test_parse_price(raw, expected):
    assert parse_price(raw) == expected


@pytest.mark.parametrize("raw", ["cheap", "", "-5000", None, True, [], "k15"])
def test_parse_price_rejects(raw):
    assert parse_price(raw) is None


def test_slots_price_conversion_and_range():
    clean, rejected = validate_slots({"max_price": "15k", "min_price": "8000"})
    assert clean["max_price"] == 15000 and clean["min_price"] == 8000 and rejected == []


def test_reversed_min_max_swapped():
    clean, rejected = validate_slots({"min_price": 20000, "max_price": 10000})
    assert (clean["min_price"], clean["max_price"]) == (10000, 20000) and rejected == []


@pytest.mark.parametrize("value", [-500, 0, 499, 500001, "2 crore"])
def test_price_out_of_range_rejected(value):
    clean, rejected = validate_slots({"max_price": value})
    assert clean["max_price"] is None and rejected == ["max_price"]


def test_price_bounds_inclusive():
    clean, _ = validate_slots({"min_price": 500, "max_price": 500000})
    assert (clean["min_price"], clean["max_price"]) == (500, 500000)


@pytest.mark.parametrize("raw,expected", [(1, 1), (5, 5), ("2", 2), ("2bhk", 2), ("3 BHK", 3), (2.0, 2)])
def test_bhk_ok(raw, expected):
    assert validate_bhk(raw) == expected


@pytest.mark.parametrize("raw", [0, 6, -1, 10, "many", True, 2.5, None])
def test_bhk_rejected(raw):
    assert validate_bhk(raw) is None


@pytest.mark.parametrize("raw,expected", [
    ("wifi", "WiFi"), ("WiFi", "WiFi"), ("Wi-Fi", "WiFi"), ("hot water", "hot-water"),
    ("geyser", "hot-water"), ("hot-water", "hot-water"), ("gym", "Gym"), ("parking", "parking"),
    ("power backup", "power-backup"), ("study table", "study-table"),
])
def test_amenity_normalised(raw, expected):
    assert validate_amenity(raw) == expected


def test_unknown_amenity_rejected():
    clean, rejected = validate_slots({"amenity": "helipad"})
    assert clean["amenity"] is None and rejected == ["amenity"]


def test_ideal_for():
    assert validate_ideal_for("student") == "Students"
    assert validate_ideal_for("working professional") == "Working Professionals"
    assert validate_ideal_for("pets") is None


def test_room_reference_must_exist():
    assert validate_room_reference(2, 3, False) == 2
    assert validate_room_reference(4, 3, False) is None
    assert validate_room_reference(0, 3, False) is None
    assert validate_room_reference(1, 0, False) is None
    assert validate_room_reference(True, 3, False) is None


def test_selected_needs_selected_room():
    assert validate_room_reference("selected", 3, True) == "selected"
    assert validate_room_reference("selected", 3, False) is None


def test_validate_slots_bad_room_reference_reported():
    clean, rejected = validate_slots({}, rooms_count=2, room_reference=5)
    assert clean["room_reference"] is None and rejected == ["room_reference"]
    clean, rejected = validate_slots({}, rooms_count=2, room_reference=2)
    assert clean["room_reference"] == 2 and rejected == []


def test_compare_references_filtered():
    clean, rejected = validate_slots({}, rooms_count=2, compare_references=[1, 2, 3])
    assert clean["compare_references"] == [1, 2] and rejected == ["compare_references"]


def test_absent_values_not_rejected():
    clean, rejected = validate_slots(Slots())
    assert rejected == [] and all(clean[k] is None for k in ("city", "bhk", "min_price", "max_price"))


def test_mixed_slots():
    clean, rejected = validate_slots(
        Slots(city="Pls Urgent", bhk=9, max_price="15k", amenity="wifi"))
    assert clean["city"] is None and clean["bhk"] is None
    assert clean["max_price"] == 15000 and clean["amenity"] == "WiFi"
    assert rejected == ["city", "bhk"]
