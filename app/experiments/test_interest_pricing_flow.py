"""
Standalone Property Interest -> Pricing API flow test.

This file tests the complete new feature in isolation:

User Query
    ↓
Property Interest Detection
    ↓
Verified Room City
    ↓
Pricing API
    ↓
Formatted Subscription Plans

It does NOT modify the existing chatbot pipeline.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


BASE_URL = "https://kangaroo-rooms-backend.onrender.com"

PRICING_PATH = (
    "/api/tenant-subscriptions/pricing"
)


CONTACT_OWNER_PHRASES = (
    "contact owner",
    "contact the owner",
    "contact landlord",
    "contact the landlord",
    "owner contact",
    "owner number",
    "owner's number",
    "landlord number",
    "landlord's number",
    "phone number",
    "contact number",
    "give me the number",
    "give me the owner's number",
    "get the number",
    "get the owner's number",
    "share the number",
    "share the owner's number",
    "can i get the owner's number",
    "how can i contact",
    "how do i contact",
    "talk to the owner",
    "talk to the landlord",
    "speak to the owner",
    "speak to the landlord",
)


PROPERTY_INTEREST_PHRASES = (
    "i want this room",
    "i want this property",
    "i want this house",
    "i want this flat",
    "i am interested",
    "i'm interested",
    "interested in this",
    "interested in the room",
    "interested in this room",
    "interested in this property",
    "i want to book",
    "want to book",
    "book this room",
    "book this property",
    "book this flat",
    "how can i book",
    "how do i book",
    "i want to visit",
    "want to visit",
    "schedule a visit",
    "schedule visit",
    "arrange a visit",
    "proceed with this room",
    "proceed with this property",
    "i want to proceed",
)


def normalize_text(
    text: str,
) -> str:

    return " ".join(
        text.lower().strip().split()
    )


def detect_property_interest(
    user_query: str,
) -> bool:

    text = normalize_text(
        user_query
    )

    if not text:

        return False

    if any(
        phrase in text
        for phrase in CONTACT_OWNER_PHRASES
    ):

        return True

    if any(
        phrase in text
        for phrase in PROPERTY_INTEREST_PHRASES
    ):

        return True

    return False


def get_subscription_pricing(
    city: str,
) -> list[dict[str, Any]] | None:

    normalized_city = city.strip().lower()

    if not normalized_city:

        print(
            "ERROR: City cannot be empty."
        )

        return None

    query = urllib.parse.urlencode(
        {
            "city": normalized_city,
        }
    )

    endpoint = (
        f"{BASE_URL}"
        f"{PRICING_PATH}"
        f"?{query}"
    )

    print("\nCalling Pricing API:")
    print(endpoint)

    request = urllib.request.Request(
        endpoint,
        method="GET",
        headers={
            "Accept": "application/json",
        },
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=30,
        ) as response:

            raw_response = response.read().decode(
                "utf-8"
            )

            data = json.loads(
                raw_response
            )

    except urllib.error.HTTPError as error:

        print(
            f"\nHTTP ERROR: {error.code}"
        )

        try:

            error_data = json.loads(
                error.read().decode(
                    "utf-8"
                )
            )

            print(
                json.dumps(
                    error_data,
                    indent=2,
                )
            )

        except Exception:

            print(
                "Could not parse API error response."
            )

        return None

    except urllib.error.URLError as error:

        print(
            f"\nCONNECTION ERROR: "
            f"{error.reason}"
        )

        return None

    except Exception as error:

        print(
            f"\nUNEXPECTED ERROR: {error}"
        )

        return None

    if not isinstance(
        data,
        dict,
    ):

        print(
            "ERROR: Invalid API response."
        )

        return None

    if data.get("success") is not True:

        print(
            "ERROR: Pricing API returned failure."
        )

        return None

    plans = data.get("data")

    if not isinstance(
        plans,
        list,
    ):

        print(
            "ERROR: Plans data is invalid."
        )

        return None

    return plans


def format_subscription_plans(
    plans: list[dict[str, Any]],
) -> str:

    if not plans:

        return (
            "No subscription plans are "
            "currently available."
        )

    lines = [
        "To unlock the property's exact location "
        "and owner contact details, you can choose "
        "one of these plans:",
        "",
    ]

    for index, plan in enumerate(
        plans,
        start=1,
    ):

        name = str(
            plan.get(
                "plan",
                "Unknown Plan",
            )
        )

        price = plan.get(
            "price",
            "N/A",
        )

        features = plan.get(
            "features",
            [],
        )

        lines.append(
            f"{index}. {name} — ₹{price}"
        )

        if isinstance(
            features,
            list,
        ):

            for feature in features:

                lines.append(
                    f"   • {feature}"
                )

        lines.append("")

    lines.append(
        "Let me know which plan you would "
        "like to choose."
    )

    return "\n".join(lines)


def handle_interest_request(
    user_query: str,
    room_city: str,
) -> str | None:
    """
    Simulates the future chatbot integration.

    Returns:
        Formatted plans if interest is detected.
        None if no interest is detected.
    """

    interest_detected = (
        detect_property_interest(
            user_query
        )
    )

    if not interest_detected:

        return None

    plans = get_subscription_pricing(
        room_city
    )

    if plans is None:

        return (
            "I couldn't retrieve the available "
            "subscription plans right now."
        )

    return format_subscription_plans(
        plans
    )


def main() -> None:

    print("=" * 70)
    print(
        "PROPERTY INTEREST -> PRICING API FLOW TEST"
    )
    print("=" * 70)

    room_city = "bangalore"

    test_queries = [

        "I want to contact the owner",

        "Can I get the owner's number?",

        "I want to book this room",

        "What amenities does this room have?",
    ]

    print(
        f"\nVerified Room City: {room_city}"
    )

    for user_query in test_queries:

        print("\n" + "=" * 70)

        print(
            f"USER QUERY: {user_query}"
        )

        print("=" * 70)

        response = handle_interest_request(
            user_query=user_query,
            room_city=room_city,
        )

        if response is None:

            print(
                "\nRESULT:"
            )

            print(
                "No property interest detected."
            )

            print(
                "Existing chatbot flow would continue."
            )

        else:

            print(
                "\nRESULT:"
            )

            print(response)


if __name__ == "__main__":

    main()