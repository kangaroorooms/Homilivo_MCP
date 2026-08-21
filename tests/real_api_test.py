import os
import json
import urllib.request


API_URL = (
    "https://homilivo.onrender.com/api/rooms"
    "?sort=latest&limit=10"
)


token = os.getenv("KANGAROO_TOKEN")

if not token:
    raise RuntimeError(
        "KANGAROO_TOKEN is not set"
    )


request = urllib.request.Request(
    API_URL,
    headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }
)


try:

    with urllib.request.urlopen(request) as response:

        print("=" * 70)
        print("REAL HOMILIVO API TEST")
        print("=" * 70)

        print(f"Status Code : {response.status}")
        print(f"API         : {API_URL}")
        print()

        data = json.loads(
            response.read().decode()
        )

        print("API RESPONSE")
        print("-" * 70)

        print(json.dumps(data, indent=2))

        print()
        print("=" * 70)
        print("REAL API TEST PASSED")
        print("=" * 70)


except Exception as e:

    print("=" * 70)
    print("REAL API TEST FAILED")
    print("=" * 70)

    print(f"Error: {e}")