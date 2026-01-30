import json
from pathlib import Path

INPUT_DIR = Path("data")
OUTPUT_DIR = Path("data_with_doc_id")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def safe_json_load(line: str, file_path: Path, line_no: int):
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        print(f"[WARN] JSON parse failed: {file_path.name} line {line_no}: {e}")
        return None

count_files = 0
count_rows = 0
count_added = 0

for fp in sorted(INPUT_DIR.glob("*.jsonl")):
    count_files += 1
    doc_id = fp.stem

    out_fp = OUTPUT_DIR / fp.name
    with fp.open("r", encoding="utf-8") as fin, out_fp.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            obj = safe_json_load(line, fp, i)
            if obj is None:
                continue

            if "doc_id" not in obj:
                obj["doc_id"] = doc_id
                count_added += 1

            if "source_file" not in obj:
                obj["source_file"] = fp.name

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count_rows += 1

print(f"Done. files={count_files}, rows={count_rows}, doc_id_added={count_added}")
print(f"Output dir: {OUTPUT_DIR.resolve()}")
