import json, glob

files = glob.glob("src/openings/eco*.json")
eco = {}
for f in files:
    eco.update(json.load(open(f)))
json.dump(eco, open("src/openings/all_openings.json", "w"), indent=2)
