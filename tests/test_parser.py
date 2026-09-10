from app.fast_parser import extract_search_parameters


def test_jaipur_budget_student_search():
    query = "Find me rooms in Jaipur under 15000 for students"

    result = extract_search_parameters(query)

    assert result["city"] == "Jaipur"
    assert result["max_price"] == 15000
    assert result["min_price"] is None
    assert "Students" in result["ideal_for"]


def test_2bhk_jaipur_search():
    query = "Find me a 2BHK in Jaipur under ₹20,000"

    result = extract_search_parameters(query)

    assert result["city"] == "Jaipur"
    assert result["max_price"] == 20000
    assert "2BHK" in result["room_types"]


def test_pg_chennai_search():
    query = "I need a PG in Chennai for working professionals"

    result = extract_search_parameters(query)

    assert result["city"] == "Chennai"
    assert "PG" in result["room_types"]
    assert "Working Professionals" in result["ideal_for"]


def test_1bhk_bangalore_search():
    query = "Find 1BHK rooms in Bangalore below 18000 for students"

    result = extract_search_parameters(query)

    assert result["city"] == "Bangalore"
    assert result["max_price"] == 18000
    assert "1BHK" in result["room_types"]
    assert "Students" in result["ideal_for"]


def test_minimum_price_udaipur_search():
    query = "Find rooms in Udaipur above 10000 for families"

    result = extract_search_parameters(query)

    assert result["city"] == "Udaipur"
    assert result["min_price"] == 10000
    assert "Family" in result["ideal_for"]


def test_default_search_parameters():
    query = "Find rooms"

    result = extract_search_parameters(query)

    assert result["city"] is None
    assert result["min_price"] is None
    assert result["max_price"] is None
    assert result["room_types"] == []
    assert result["ideal_for"] == []
    assert result["sort"] == "latest"
    assert result["limit"] == 10


# ============================================================
# BUG 1 - 'k' price shorthand
# ============================================================

def test_k_shorthand_max_price():
    result = extract_search_parameters("2bhk under 15k in Bangalore")
    assert result["max_price"] == 15000


def test_k_shorthand_min_price():
    result = extract_search_parameters("rooms above 8k in Pune")
    assert result["min_price"] == 8000


def test_k_shorthand_various_magnitudes():
    assert extract_search_parameters("under 18k")["max_price"] == 18000
    assert extract_search_parameters("under 8k")["max_price"] == 8000
    assert extract_search_parameters("under 20k")["max_price"] == 20000


def test_plain_number_without_k_is_unaffected():
    result = extract_search_parameters("3 bhk under 15")
    assert result["max_price"] == 15


# ============================================================
# BUG 2 - city alias resolution + multi-word localities
# ============================================================

def test_blr_alias_resolves_to_bangalore():
    result = extract_search_parameters("rooms in blr under 20k")
    assert result["city"] == "Bangalore"


def test_hsr_alias_resolves_to_hsr_layout():
    result = extract_search_parameters("2bhk in HSR for a family")
    assert result["city"] == "HSR Layout"


def test_multiword_locality_is_not_truncated():
    result = extract_search_parameters("2bhk flat in Electronic City for a family")
    assert result["city"] == "Electronic City"


# ============================================================
# BUG 3 - "between X and Y" price ranges
# ============================================================

def test_between_price_range():
    result = extract_search_parameters(
        "rooms in Kolkata between 15000 and 25000"
    )
    assert result["min_price"] == 15000
    assert result["max_price"] == 25000


def test_between_price_range_with_k_shorthand():
    result = extract_search_parameters("rooms between 15k and 25k")
    assert result["min_price"] == 15000
    assert result["max_price"] == 25000


# ============================================================
# BUG 4 - spaced room types (N BHK / N RK)
# ============================================================

def test_spaced_bhk_is_normalized():
    result = extract_search_parameters("3 bhk under 15000")
    assert result["room_types"] == ["3BHK"]


def test_spaced_rk_is_normalized():
    result = extract_search_parameters("1 rk near Mansarovar")
    assert result["room_types"] == ["1RK"]


def test_concatenated_bhk_still_works():
    result = extract_search_parameters("2BHK in Bangalore")
    assert result["room_types"] == ["2BHK"]


def test_any_n_bhk_is_generalized():
    result = extract_search_parameters("5 bhk in Mumbai")
    assert result["room_types"] == ["5BHK"]