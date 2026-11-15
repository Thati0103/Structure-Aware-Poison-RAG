"""CSV validation and experiment construction. No model/database imports.

Set random_controls=false in config.json to explicitly omit Experiment A's A4.
The default is true, requiring matched random-edit annotations and PDFs.
"""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

ATTACKS = ("cell", "header", "chart", "layout")
ATTACK_CONDITIONS = {
    "cell": "a2_cell", "header": "a3_header", "chart": "a5_chart", "layout": "a6_layout",
}
COLUMNS = [
    "question_id", "doc_name", "question", "support_pages", "clean_answer",
    "cell_doc_name", "cell_edited_page", "cell_target_answer",
    "header_doc_name", "header_edited_page", "header_target_answer",
    "random_cell_doc_name", "random_cell_edited_page",
    "random_header_doc_name", "random_header_edited_page",
]
COLUMNS += [
    f"{attack}_{field}"
    for attack in ("chart", "layout")
    for field in ("doc_name", "edited_page", "target_answer")
]
COLUMNS += [
    f"random_{attack}_{field}"
    for attack in ("chart", "layout")
    for field in ("doc_name", "edited_page")
]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as pdf_file:
        for chunk in iter(lambda: pdf_file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["root"] = str(path.parent)
    cfg.setdefault("duplicate_dir", "data/docs/duplicates")
    cfg.setdefault("questions_file", "data/questions.jsonl")
    cfg.setdefault("attacks_file", "data/attacks.jsonl")
    cfg.setdefault("state_dir", "results/_state")
    cfg.setdefault("corpus_warning_multiplier", 5)
    for key in ("clean_dir", "duplicate_dir", "csv", "questions_file", "attacks_file", "state_dir", "results_dir"):
        cfg[key] = str((path.parent / cfg[key]).resolve())
    cfg["variant_dirs"] = {
        attack: str((path.parent / folder).resolve())
        for attack, folder in cfg["variant_dirs"].items()
    }
    for key in ("retrieval_k", "context_pages", "dpi", "max_new_tokens", "corpus_warning_multiplier"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["context_pages"] > cfg["retrieval_k"]:
        raise ValueError("context_pages cannot exceed retrieval_k")
    return cfg


def resolve_pdf(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.suffix.lower() != ".pdf":
        raise ValueError(f"Missing PDF or invalid relative filename: {name} in {root}")
    return str(path)


def page_numbers(text):
    try:
        pages = [int(p.strip()) for p in text.split(";")]
    except ValueError as exc:
        raise ValueError(f"Use PDF page numbers separated by semicolons: {text!r}") from exc
    if not pages or min(pages) < 1 or len(pages) != len(set(pages)):
        raise ValueError(f"Invalid or repeated support pages: {text!r}")
    return pages


def attack_fields(attack, random_controls=True):
    fields = (f"{attack}_doc_name", f"{attack}_edited_page", f"{attack}_target_answer")
    if random_controls:
        fields += (f"random_{attack}_doc_name", f"random_{attack}_edited_page")
    return fields


def row_attacks(row, enabled):
    """Blank attack columns mean that attack does not apply to this question."""
    return [attack for attack in enabled if any(row.get(field) for field in attack_fields(attack))]


def visual_page_hashes(path, dpi, cache):
    import pymupdf

    if path not in cache:
        with pymupdf.open(path) as pdf:
            cache[path] = [
                hashlib.sha256(page.get_pixmap(dpi=dpi, alpha=False).samples).hexdigest()
                for page in pdf
            ]
    return cache[path]


def validate_required_fields(row, columns, attacks, random_controls, line):
    required = set(COLUMNS[:5])
    for attack in attacks:
        required.update(attack_fields(attack, random_controls))
    missing_columns = required - columns
    if missing_columns:
        raise ValueError(f"CSV row {line}: missing columns {sorted(missing_columns)}")
    empty_values = sorted(field for field in required if not row.get(field))
    if empty_values:
        raise ValueError(f"CSV row {line}: empty required values {empty_values}")


def validate_variant(cfg, row, prefix, original_page_count, verify_edits, visual_cache):
    import pymupdf

    question_id = row["question_id"]
    try:
        edited_page = int(row[f"{prefix}_edited_page"])
    except ValueError as exc:
        raise ValueError(f"{question_id}: invalid {prefix} edited page") from exc
    if edited_page not in row["support"]:
        raise ValueError(f"{question_id}: {prefix} edited page must be a supporting page")

    variant_path = resolve_pdf(cfg["variant_dirs"][prefix], row[f"{prefix}_doc_name"])
    row[f"{prefix}_path"] = variant_path
    row[f"{prefix}_edited_page"] = edited_page
    with pymupdf.open(variant_path) as pdf:
        if len(pdf) != original_page_count:
            raise ValueError(f"{question_id}: {prefix} must retain original page count")

    if verify_edits:
        original_hashes = visual_page_hashes(row["clean_path"], cfg["dpi"], visual_cache)
        variant_hashes = visual_page_hashes(variant_path, cfg["dpi"], visual_cache)
        changed_pages = [
            page_num for page_num, (original, variant)
            in enumerate(zip(original_hashes, variant_hashes), start=1)
            if original != variant
        ]
        if changed_pages != [edited_page]:
            raise ValueError(
                f"{question_id}: {prefix} visual changes on {changed_pages}; "
                f"expected only [{edited_page}]"
            )


def validate_question_pdfs(cfg, row, attacks, verify_edits, visual_cache):
    import pymupdf

    row["clean_path"] = resolve_pdf(cfg["clean_dir"], row["doc_name"])
    row["support"] = page_numbers(row["support_pages"])
    with pymupdf.open(row["clean_path"]) as pdf:
        original_page_count = len(pdf)
    if max(row["support"]) > original_page_count:
        raise ValueError(f"{row['question_id']}: support page exceeds PDF length")

    for attack in attacks:
        validate_variant(cfg, row, attack, original_page_count, verify_edits, visual_cache)
        if row["_random_controls"]:
            random_prefix = f"random_{attack}"
            validate_variant(cfg, row, random_prefix, original_page_count, verify_edits, visual_cache)
            if row[f"{attack}_edited_page"] != row[f"{random_prefix}_edited_page"]:
                raise ValueError(
                    f"{row['question_id']}: matched random edit must use the same page as {attack}"
                )

    if {"cell", "header"}.issubset(attacks):
        if row["cell_edited_page"] != row["header_edited_page"]:
            raise ValueError(f"{row['question_id']}: matched cell/header attacks must edit the same page")


def load_questions(cfg, attacks=(), verify_edits=False):
    random_controls = cfg.get("random_controls", True)
    if type(random_controls) is not bool:
        raise ValueError("random_controls must be true or false in config.json")

    rows = []
    seen_question_ids = set()
    visual_cache = {}
    with Path(cfg["csv"]).open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = set(reader.fieldnames or [])
        missing_columns = set(COLUMNS[:5]) - columns
        if missing_columns:
            raise ValueError(f"Missing CSV columns: {sorted(missing_columns)}")

        for line, raw_row in enumerate(reader, start=2):
            if None in raw_row:
                raise ValueError(f"CSV row {line}: too many fields; quote answers containing commas")
            row = {field: (value or "").strip() for field, value in raw_row.items()}
            selected = row_attacks(row, attacks)
            validate_required_fields(row, columns, selected, random_controls, line)
            row["_random_controls"] = random_controls
            if row["question_id"] in seen_question_ids:
                raise ValueError(f"Duplicate question_id: {row['question_id']}")
            seen_question_ids.add(row["question_id"])
            validate_question_pdfs(cfg, row, selected, verify_edits, visual_cache)
            rows.append(row)

    if not rows:
        raise ValueError("CSV has no question rows")
    return rows


def ensure_duplicate(clean_path, doc_name, duplicate_dir):
    duplicate_root = Path(duplicate_dir).resolve()
    duplicate_path = (duplicate_root / doc_name).resolve()
    if not duplicate_path.is_relative_to(duplicate_root):
        raise ValueError("Invalid duplicate filename")
    duplicate_path.parent.mkdir(parents=True, exist_ok=True)
    if duplicate_path == Path(clean_path):
        raise ValueError("duplicate_dir must be separate from clean_dir")
    if not duplicate_path.exists() or file_hash(duplicate_path) != file_hash(clean_path):
        temporary = duplicate_path.with_suffix(".copying")
        shutil.copyfile(clean_path, temporary)
        temporary.replace(duplicate_path)
    return str(duplicate_path)


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def prepare_dataset(cfg, attacks=(), verify_edits=False):
    """Validate questions, make unedited copies, and save their annotations."""
    rows = load_questions(cfg, attacks, verify_edits)
    questions = []
    edits = []
    for row in rows:
        row["duplicate_path"] = ensure_duplicate(
            row["clean_path"], row["doc_name"], cfg["duplicate_dir"]
        )
        questions.append({
            "question_id": row["question_id"],
            "doc_name": row["doc_name"],
            "question": row["question"],
            "support_pages": row["support"],
            "clean_answer": row["clean_answer"],
        })
        for attack in row_attacks(row, attacks):
            edit = {
                "question_id": row["question_id"],
                "attack_type": attack,
                "doc_name": row[f"{attack}_doc_name"],
                "edited_page": row[f"{attack}_edited_page"],
                "target_answer": row[f"{attack}_target_answer"],
                "random_controls": row["_random_controls"],
            }
            if row["_random_controls"]:
                edit["random_doc_name"] = row[f"random_{attack}_doc_name"]
                edit["random_edited_page"] = row[f"random_{attack}_edited_page"]
            edits.append(edit)
    write_jsonl(cfg["questions_file"], questions)
    write_jsonl(cfg["attacks_file"], edits)
    return rows


def page(path, number, origin, slot, poisoned=False, edited=False):
    return {"filepath": str(path), "page_num": number, "origin_doc": origin,
            "slot": slot, "page_id": f"{slot}:{origin}:{number}",
            "document_id": digest([slot, origin])[:12], "ispoisoned": poisoned,
            "is_edited": edited}


def clean_corpus(cfg):
    import pymupdf
    pages = []
    for path in sorted(candidate for candidate in Path(cfg["clean_dir"]).rglob("*") if candidate.is_file() and candidate.suffix.lower() == ".pdf"):
        name = path.relative_to(cfg["clean_dir"]).as_posix()
        with pymupdf.open(path) as pdf:
            pages.extend(page(path.resolve(), number, name, "clean") for number in range(1, len(pdf) + 1))
    if not pages:
        raise ValueError("No clean PDFs found")
    return pages


def make_a_jobs(rows, clean_pages, attacks, conditions=None):
    """Add one full document copy to the shared clean corpus for each A condition."""
    jobs = []
    clean_doc_names = {page_info["origin_doc"] for page_info in clean_pages}
    for row in rows:
        if row["doc_name"] not in clean_doc_names:
            raise ValueError(f"Clean corpus does not contain {row['doc_name']}")
        original_pages = [
            page_info for page_info in clean_pages if page_info["origin_doc"] == row["doc_name"]
        ]
        selected_attacks = row_attacks(row, attacks)
        variants = [("a0", "", ""), ("a1", "", "")]
        for attack in selected_attacks:
            variants.append((ATTACK_CONDITIONS[attack], attack, attack))
            if row.get("_random_controls", True):
                variants.append((f"a4_random_{attack}", f"random_{attack}", ""))

        for condition, variant_prefix, attack in variants:
            if conditions and condition not in conditions:
                continue
            added_pages = []
            tracked_pages = {}
            if condition != "a0":
                if variant_prefix:
                    variant_path = row[f"{variant_prefix}_path"]
                    edited_page = row[f"{variant_prefix}_edited_page"]
                else:
                    variant_path = row["duplicate_path"]
                    edited_page = None
                for original in original_pages:
                    page_num = original["page_num"]
                    is_edited = page_num == edited_page
                    added_pages.append(page(
                        variant_path, page_num, row["doc_name"], "added",
                        poisoned=bool(attack and is_edited), edited=is_edited,
                    ))

                if condition == "a1":
                    tracked_pages = {
                        name: row[f"{name}_edited_page"] for name in selected_attacks
                    }
                elif variant_prefix:
                    tracked_pages = {variant_prefix.removeprefix("random_"): edited_page}

            baseline = "a1" if attack or condition.startswith("a4_") else None
            jobs.append({
                "experiment": "A", "condition": condition, "row": row,
                "attack": attack, "baseline": baseline,
                "tracked_pages": tracked_pages, "pages": clean_pages + added_pages,
            })
    return jobs


def make_b_jobs(rows, attacks, conditions=None):
    """Supply the generator with the exact page list required by each B condition."""
    jobs = []
    for row in rows:
        clean_pages = [
            page(row["clean_path"], number, row["doc_name"], "clean")
            for number in row["support"]
        ]
        contexts = [("b0", [], "", None), ("b1", clean_pages, "", None)]
        for attack in row_attacks(row, attacks):
            target_page_num = row[f"{attack}_edited_page"]
            target_clean = next(
                item for item in clean_pages if item["page_num"] == target_page_num
            )
            other_support = [
                item for item in clean_pages if item["page_num"] != target_page_num
            ]
            duplicate = page(row["duplicate_path"], target_page_num, row["doc_name"], "added")
            poisoned = page(
                row[f"{attack}_path"], target_page_num, row["doc_name"], "added",
                poisoned=True, edited=True,
            )
            replacement = []
            for original in clean_pages:
                if original["page_num"] == target_page_num:
                    replacement.append(page(
                        row[f"{attack}_path"], target_page_num, row["doc_name"], "clean",
                        poisoned=True, edited=True,
                    ))
                else:
                    replacement.append(dict(original))

            contexts.extend([
                (f"b2_{attack}", replacement, attack, "b1"),
                (f"b3_{attack}", [target_clean, duplicate] + other_support, "", None),
                (f"b4_{attack}_clean_first", [target_clean, poisoned] + other_support, attack, f"b3_{attack}"),
                (f"b4_{attack}_poison_first", [poisoned, target_clean] + other_support, attack, f"b3_{attack}"),
            ])
        for condition, pages, attack, baseline in contexts:
            if conditions and condition not in conditions:
                continue
            jobs.append({
                "experiment": "B", "condition": condition, "row": row,
                "attack": attack, "baseline": baseline, "pages": pages,
            })
    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().with_name("config.json")))
    parser.add_argument("--attacks", nargs="*", choices=ATTACKS, default=[])
    parser.add_argument("--verify-edits", action="store_true")
    args = parser.parse_args()
    rows = prepare_dataset(read_config(args.config), args.attacks, args.verify_edits)
    print(f"OK: {len(rows)} questions; {len({r['doc_name'] for r in rows})} source documents")
    print("Page numbers are 1-based PDF page positions. Supporting-page annotations need manual verification.")
    print("Generated questions.jsonl, attacks.jsonl, and unedited PDFs in duplicates/.")
