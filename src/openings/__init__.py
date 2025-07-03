import json, os, random

def load_opening_fens(path="src/openings/all_openings.json", max_samples=500):
    if not os.path.exists(path):
        print(f"[Warning] Opening ECO JSON not found at {path}")
        return []

    eco = json.load(open(path))
    fens = [v.get("fen") or v.get("epd") for v in eco.values() if v.get("fen") or v.get("epd")]
    return random.sample(fens, min(max_samples, len(fens)))
