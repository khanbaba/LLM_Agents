import requests
import json
import time

location_ids = ['p26', 'p23', 'p24']
all_room_details = []
pages_to_fetch = 5

for location_id in location_ids:
    print(f"Processing location ID {location_id}...")

    for page in range(1, pages_to_fetch + 1):
        print(f"  Fetching page {page}...")

        # Step 1: Fetch search results
        search_url = "https://api.jajiga.com/api/search"
        params = {
            "per_page": 18,
            "page": page,
            "locations[]": location_id,
            "without[]": "map"
        }

        search_response = requests.get(search_url, params=params)

        if search_response.status_code != 200:
            print(f"    Failed to fetch search results for page {page}: {search_response.status_code}")
            continue

        search_data = search_response.json()
        room_items = search_data.get('rooms', {}).get('items', [])

        if not room_items:
            print(f"    No rooms found on page {page}. Stopping early for this location.")
            break

        # Step 2: Fetch room details
        for room in room_items:
            room_id = room.get('id')
            if room_id:
                detail_url = f"https://api.jajiga.com/api/room/{room_id}"
                detail_response = requests.get(detail_url)

                if detail_response.status_code == 200:
                    room_data = detail_response.json()
                    room_data['location_id'] = location_id  # Tag with location
                    room_data['page'] = page                # Tag with page number
                    all_room_details.append(room_data)
                    print(f"    ✓ Fetched room ID {room_id}")
                else:
                    print(f"    ✗ Failed to fetch details for room ID {room_id}: {detail_response.status_code}")

                time.sleep(0.5)  # API friendly pause

# Step 3: Save all room details
with open("room_details.json", "w", encoding='utf-8') as f:
    json.dump(all_room_details, f, ensure_ascii=False, indent=2)

print("✅ All room details saved to room_details.json")
