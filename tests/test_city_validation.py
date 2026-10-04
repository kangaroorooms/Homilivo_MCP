"""City validation in fast_parser: garbage never becomes a city."""

import pytest

from app.experiments.agent_modes import merge_search_parameters
from app.fast_parser import _validate_city, extract_search_parameters


def _city(query):
    return extract_search_parameters(query)["city"]


@pytest.mark.parametrize(
    "query, expected",
    [
        ("something in a different area", None),
        ("i want i want a room room in bangalor pls under under 15000", "Bangalore"),
        ("rroom in bangalr under", "Bangalore"),
    ],
)
def test_garbage_is_cleaned_or_rejected(query, expected):
    assert _city(query) == expected


@pytest.mark.parametrize(
    "query, expected",
    [
        ("rooms in vadodara", "Vadodara"),
        ("flats in nagpur", "Nagpur"),
        ("rooms in new delhi", "New Delhi"),
        ("show me rooms in navi mumbai", "Navi Mumbai"),
        ("hydrabad mein PG chahiye", "Hyderabad"),
        ("bangalre mein room chahiye", "Bangalore"),
        ("rooms in jaipur", "Jaipur"),
        ("rooms in udaipur", "Udaipur"),
    ],
)
def test_legitimate_cities_unchanged(query, expected):
    assert _city(query) == expected


@pytest.mark.parametrize(
    "name", ["Nagpur", "Vadodara", "Mysore", "Surat", "Indore", "Pune", "Pimpri"]
)
def test_fuzzy_never_maps_real_city_to_another(name):
    assert _city(f"rooms in {name}") == name
    assert _city(f"any rooms {name}") == name


def test_validate_city_rejects_long_and_empty():
    assert _validate_city("a b c d") is None
    assert _validate_city("pls urgent") is None
    assert _validate_city("") is None


def test_refinement_keeps_previous_city_when_validation_gives_none():
    previous = {"city": "Bangalore", "max_price": 20000, "sort": "latest", "limit": 10}
    new = extract_search_parameters("something in a different area under 15000")
    assert new["city"] is None
    merged = merge_search_parameters(previous, new)
    assert merged["city"] == "Bangalore"
    assert merged["max_price"] == 15000


def test_before_after_table(capsys):
    before = {
        "something in a different area": "A Different Area",
        "i want i want a room room in bangalor pls under under 15000": "Bangalor Pls",
        "rroom in bangalr under": "Bangalr",
        "rooms in vadodara": "Vadodara",
        "pg near manipal university": None,
        "flats in nagpur": "Nagpur",
        "bhopal me 1bhk kiraye pe chahiye": None,
        "rooms in new delhi": "New Delhi",
        "show me rooms in navi mumbai": "Navi Mumbai",
        "hydrabad mein PG chahiye": "Hyderabad",
        "bangalre mein room chahiye": "Bangalore",
    }
    with capsys.disabled():
        print("\n%-62s %-18s %s" % ("query", "before", "after"))
        for q, b in before.items():
            print("%-62s %-18r %r" % (q, b, _city(q)))
