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