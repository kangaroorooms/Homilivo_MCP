"""
Standalone property interest detection test.

This file is isolated from the chatbot pipeline.

It determines whether a user message indicates
strong interest in a property and may require
showing subscription plans.

It does NOT modify:
- agent_modes.py
- LangGraph
- Guardrails
- Existing routing
- Room search pipeline
"""


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
    """
    Normalize user text for phrase matching.
    """

    return " ".join(
        text.lower().strip().split()
    )


def detect_property_interest(
    user_query: str,
) -> bool:
    """
    Return True only when the user shows
    strong action intent towards a property
    or explicitly requests contact information.
    """

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


def main() -> None:

    test_queries = [

        # SHOULD TRIGGER

        "I want to contact the owner",

        "Can I get the owner's number?",

        "Please share the contact number",

        "How can I contact the owner?",

        "Can I talk to the owner?",

        "I am interested in this room",

        "I want this property",

        "I want to book this room",

        "Can I schedule a visit?",

        "I want to proceed with this property",


        # SHOULD NOT TRIGGER

        "I really like this flat",

        "What is the rent?",

        "Does this room have AC?",

        "What amenities are available?",

        "Show me rooms in Bangalore",

        "What is the location?",

        "Tell me more about this room",

        "Compare these rooms",

        "Show the rooms in a table",
    ]


    print("=" * 70)

    print(
        "PROPERTY INTEREST DETECTION TEST"
    )

    print("=" * 70)


    for query in test_queries:

        result = (
            detect_property_interest(
                query
            )
        )

        status = (
            "INTEREST DETECTED"
            if result
            else "NO INTEREST"
        )

        print()

        print(
            query
        )

        print(
            f"-> {status}"
        )


if __name__ == "__main__":

    main()