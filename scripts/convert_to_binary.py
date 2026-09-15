import yaml
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1] / "datasets" / "archive"
OUT_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "classify_binary"

with open(ROOT / "data.yaml", "r") as f:
    data = yaml.safe_load(f)
    names = data.get("names")

if names is None:
    raise SystemExit("Could not read class names from data.yaml")

try:
    HEALTHY_IDX = names.index("Healthy")
except ValueError:
    raise SystemExit("'Healthy' class not found in data.yaml names")

splits = ["train", "valid", "test"]
count = {s: {"Healthy": 0, "Diseased": 0} for s in splits}

for split in splits:
    images_dir = ROOT / split / "images"
    labels_dir = ROOT / split / "labels"
    out_split = OUT_ROOT / split
    (out_split / "Healthy").mkdir(parents=True, exist_ok=True)
    (out_split / "Diseased").mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        print(f"Skipping missing split: {split}")
        continue

    for img_path in images_dir.glob("*"):
        if not img_path.is_file():
            continue
        stem = img_path.stem
        lbl = labels_dir / (stem + ".txt")
        if not lbl.exists():
            # no label file; skip this image
            continue
        with open(lbl, "r") as lf:
            classes = [int(line.split()[0]) for line in lf.read().splitlines() if line.strip()]
        if not classes:
            continue
        if all(c == HEALTHY_IDX for c in classes):
            dst = out_split / "Healthy" / img_path.name
            shutil.copy2(img_path, dst)
            count[split]["Healthy"] += 1
        else:
            dst = out_split / "Diseased" / img_path.name
            shutil.copy2(img_path, dst)
            count[split]["Diseased"] += 1

print("Conversion complete. Summary:")
for s in splits:
    print(f"{s}: Healthy={count[s]['Healthy']}, Diseased={count[s]['Diseased']}")
