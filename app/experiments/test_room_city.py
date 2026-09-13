"""
Standalone room city extraction test.

This does not modify the chatbot pipeline.

It tests safely extracting a city from the
verified room data structure.
"""

from typing import Any


def get_room_city(
    room: dict[str, Any],
) -> str | None:
    """
    Extract city from verified room data.

    Priority:
    1. Direct city field
    2. Nested location city
    3. Combined location containing a comma

    Returns None when a city cannot be safely
    determined.
    """

    city = room.get(
        "city"
    )

    if isinstance(
        city,
        str,
    ) and city.strip():

        return city.strip().lower()


    location = room.get(
        "location"
    )

    if isinstance(
        location,
        dict,
    ):

        nested_city = location.get(
            "city"
        )

        if isinstance(
            nested_city,
            str,
        ) and nested_city.strip():

            return nested_city.strip().lower()


    if isinstance(
        location,
        str,
    ) and "," in location:

        parts = [
            part.strip()
            for part in location.split(",")
            if part.strip()
        ]

        if len(parts) >= 2:

            return parts[-1].lower()


    return None


def main() -> None:

    test_rooms = [

        {
            "title": "Room With City",
            "location": (
                "Koramangala"
            ),
            "city": "Bangalore",
        },

        {
            "title": "Room With Combined Location",
            "location": (
                "Koramangala, Bangalore"
            ),
        },

        {
            "title": "Room With Nested Location",
            "location": {
                "area": "Koramangala",
                "city": "Bangalore",
            },
        },

        {
            "title": "Room Without City",
            "location": "Koramangala",
        },

        {
            "title": "Room With Empty Location",
            "location": "",
        },

    ]


    print("=" * 70)

    print(
        "SAFE ROOM CITY EXTRACTION TEST"
    )

    print("=" * 70)


    for room in test_rooms:

        city = get_room_city(
            room
        )

        print()

        print(
            f"Room: {room['title']}"
        )

        print(
            f"Extracted City: {city}"
        )


if __name__ == "__main__":

    main()