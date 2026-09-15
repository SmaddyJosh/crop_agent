import yaml
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1] / "datasets" / "archive"
OUT_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "classify_multiclass"

with open(ROOT / "data.yaml", "r") as f:
    data = yaml.safe_load(f)
    names = data.get("names")

if names is None:
    raise SystemExit("Could not read class names from data.yaml")

# sanitize folder names
def safe_name(s):
    return s.replace(" ", "_").replace('/', '_')

splits = ["train", "valid", "test"]
# counts[split][class_name]
counts = {s: {safe_name(n): 0 for n in names} for s in splits}

for split in splits:
    images_dir = ROOT / split / "images"
    labels_dir = ROOT / split / "labels"
    out_split = OUT_ROOT / split

    # create class folders
    for n in names:
        (out_split / safe_name(n)).mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        print(f"Skipping missing split: {split}")
        continue

    for img_path in images_dir.glob("*"):
        if not img_path.is_file():
            continue
        stem = img_path.stem
        lbl = labels_dir / (stem + ".txt")
        if not lbl.exists():
            continue
        with open(lbl, "r") as lf:
            lines = [l.strip().split() for l in lf.read().splitlines() if l.strip()]
        if not lines:
            continue
        # choose class with largest bbox area (w*h) in YOLO normalized coords
        best_cls = None
        best_area = -1.0
        for parts in lines:
            try:
                c = int(parts[0])
                w = float(parts[3])
                h = float(parts[4])
                area = w * h
            except Exception:
                continue
            if area > best_area:
                best_area = area
                best_cls = c
        if best_cls is None:
            continue
        cls_name = safe_name(names[best_cls])
        dst = out_split / cls_name / img_path.name
        # avoid overwriting; if file exists, append a suffix
        if dst.exists():
            base = dst.stem
            suffix = 1
            while True:
                newname = f"{base}_{suffix}{dst.suffix}"
                newdst = dst.with_name(newname)
                if not newdst.exists():
                    dst = newdst
                    break
                suffix += 1
        shutil.copy2(img_path, dst)
        counts[split][cls_name] += 1

print("Multiclass conversion complete. Summary:")
for s in splits:
    print(f"\n{s}:")
    total = 0
    for n in names:
        sn = safe_name(n)
        c = counts[s][sn]
        total += c
        print(f"  {sn}: {c}")
    print(f"  total images assigned: {total}")
