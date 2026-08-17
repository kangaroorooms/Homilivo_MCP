from fast_parser import extract_search_parameters


test_queries = [

    "Find me rooms in Jaipur under 15000 for students",

    "Find me a 2BHK in Jaipur under ₹20,000",

    "I need a PG in Chennai for working professionals",

    "Find 1BHK rooms in Bangalore below 18000 for students",

    "Find rooms in Udaipur above 10000 for families"
]


print("=" * 70)
print("              FAST PARAMETER PARSER TEST")
print("=" * 70)


for query in test_queries:

    print("\nUSER REQUEST")
    print("-" * 70)
    print(query)

    result = extract_search_parameters(query)

    print("\nEXTRACTED PARAMETERS")
    print("-" * 70)

    for key, value in result.items():

        print(f"{key:15}: {value}")


print("\n" + "=" * 70)
print("                 TEST COMPLETED")
print("=" * 70)