
from pathlib import Path
import requests

URL = "https://raw.githubusercontent.com/Raffaele2004/netflix-data-analysis/main/netflix_movies_detailed_up_to_2025.csv"
OUT = Path(__file__).parent / "data" / "netflix_movies_detailed_up_to_2025.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

print("Downloading Netflix movie dataset...")
r = requests.get(URL, timeout=60)
r.raise_for_status()
OUT.write_bytes(r.content)
print(f"Saved: {OUT}")
