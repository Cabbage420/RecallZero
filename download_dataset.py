
import json
from pathlib import Path
from urllib.request import urlretrieve

BASE_URL = "https://huggingface.co/datasets/locuslab/TOFU/resolve/main/"
DATA_DIR = Path("data/raw")

DATA_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "full.json",
    "forget01.json",
    "retain99.json",
]

for filename in FILES:
    destination = DATA_DIR / filename

    if not destination.exists():
        print(f"Downloading {filename}...")
        urlretrieve(BASE_URL + filename, destination)

    # TOFU stores one JSON object per line (JSONL).
    with destination.open("r", encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    assert all(
        "question" in row and "answer" in row
        for row in records
    ), f"Invalid dataset format: {filename}"

    print(f"{filename}: {len(records)} records")

    if filename == "forget01.json":
        print("\nExample target for unlearning:")
        print("Question:", records[0]["question"])
        print("Answer:", records[0]["answer"])

print("\nDataset preparation complete.")
