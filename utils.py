import pandas as pd
import re
import random

# ---------- Default Amenities ----------
DEFAULT_AMENITIES = ["WiFi", "Kitchen", "Parking", "Air Conditioning", "TV", "Workspace", "Gym", "Pool"]

# ---------- Helper Functions ----------
def extract_bhk(title):
    """Extract number of bedrooms from 'X BHK'."""
    match = re.search(r'(\d+)\s*BHK', title, re.IGNORECASE)
    return int(match.group(1)) if match else None


def extract_property_type(title):
    """Detect property type from title text."""
    title_lower = title.lower()
    if "villa" in title_lower:
        return "villa"
    elif "independent house" in title_lower:
        return "independent house"
    elif "house" in title_lower:
        return "house"
    elif "studio" in title_lower:
        return "studio"
    elif "flat" in title_lower or "apartment" in title_lower:
        return "apartment"
    elif "penthouse" in title_lower:
        return "penthouse"
    elif "plot" in title_lower or "land" in title_lower:
        return "plot"
    elif "bungalow" in title_lower:
        return "bungalow"
    else:
        return "other"


def extract_city_state(location):
    """Extract city & state from Location string."""
    parts = [p.strip() for p in location.split(",")]
    if len(parts) >= 3:
        city = parts[-2]
        state = parts[-1]
    elif len(parts) == 2:
        city = parts[0]
        state = parts[1]
    else:
        city = parts[0]
        state = "Unknown"
    return city, state


def infer_amenities(desc):
    """Infer amenities from description & ensure at least 4."""
    amenities = []
    desc_lower = desc.lower()
    if "school" in desc_lower or "college" in desc_lower:
        amenities.append("Workspace")
    if "hospital" in desc_lower:
        amenities.append("Air Conditioning")
    if "pool" in desc_lower:
        amenities.append("Pool")
    if "garden" in desc_lower:
        amenities.append("Gym")

    # Fill to 4 minimum
    while len(amenities) < 4:
        choice = random.choice(DEFAULT_AMENITIES)
        if choice not in amenities:
            amenities.append(choice)
    return amenities


# ---------- Main Function ----------
def enrich_properties(data: list) -> list:
    """Take raw property data list and return enriched list."""
    enriched = []
    for item in data:
        city, state = extract_city_state(item["Location"])
        bedrooms = extract_bhk(item["Property Title"])
        prop_type = extract_property_type(item["Property Title"])
        guests = bedrooms * 2 if bedrooms else None

        enriched.append({
            "title": item["Property Title"],
            "description": item["Description"],
            "property_type": prop_type,
            "address": item["Location"],
            "city": city,
            "state": state,
            "country": "India",
            "postal_code": None,   # placeholder, since not available
            "price": item["Price"],
            "bedrooms": bedrooms,
            "bathrooms": item["Baths"],
            "guests": guests,
            "amenities": infer_amenities(item["Description"])
        })
    return enriched
