import re

def extract_vessel_name(query: str) -> str:
    # This regex will match capitalized or mixed-case vessel names
    match = re.search(r'\b[A-Za-z][A-Za-z0-9 ]+\b', query)
    return match.group(0) if match else ''


def clean_vessel_name(vessel_name: str) -> str:
    # Normalize the vessel name (if needed)
    return vessel_name.strip().upper()
