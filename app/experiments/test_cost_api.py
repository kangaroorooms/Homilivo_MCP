"""
Standalone Tenant Subscription Pricing API test.

This file is completely isolated from the chatbot pipeline.

It:
1. Calls the subscription pricing API
2. Validates the API response
3. Extracts available plans
4. Formats plans for chatbot display

It does NOT modify:
- agent_modes.py
- LangGraph
- Guardrails
- Existing routing
- Room search pipeline
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


def get_subscription_pricing(
    city: str,
) -> list[dict[str, Any]] | None:
    """
    Fetch subscription pricing for a city.

    Returns:
        List of plans on success.
        None if the API call fails.
    """

    normalized_city = city.strip().lower()

    if not normalized_city:

        print("ERROR: City cannot be empty.")

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

    except json.JSONDecodeError:

        print(
            "\nERROR: API returned invalid JSON."
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
            "\nERROR: Invalid API response format."
        )

        return None

    if data.get("success") is not True:

        print(
            "\nERROR: API returned success=False"
        )

        print(
            json.dumps(
                data,
                indent=2,
            )
        )

        return None

    plans = data.get(
        "data"
    )

    if not isinstance(
        plans,
        list,
    ):

        print(
            "\nERROR: Plans data is missing "
            "or invalid."
        )

        return None

    return plans


def format_subscription_plans(
    plans: list[dict[str, Any]],
) -> str:
    """
    Format verified subscription plans
    into a chatbot-friendly response.
    """

    if not plans:

        return (
            "No subscription plans are "
            "currently available."
        )

    lines = [
        "To unlock property contact details, "
        "you can choose one of these plans:",
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
        "Let me know which plan you "
        "would like to choose."
    )

    return "\n".join(
        lines
    )


def main() -> None:

    print("=" * 70)
    print("TENANT SUBSCRIPTION PRICING TEST")
    print("=" * 70)

    city = "bangalore"

    print(
        f"\nTesting city: {city}"
    )

    plans = get_subscription_pricing(
        city
    )

    if plans is None:

        print(
            "\nPricing API test failed."
        )

        return

    print("\n" + "=" * 70)
    print("VERIFIED PLANS")
    print("=" * 70)

    print(
        json.dumps(
            plans,
            indent=2,
        )
    )

    formatted_response = (
        format_subscription_plans(
            plans
        )
    )

    print("\n" + "=" * 70)
    print("CHATBOT RESPONSE PREVIEW")
    print("=" * 70)

    print(
        formatted_response
    )


if __name__ == "__main__":

    main()