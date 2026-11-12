"""Three visual RAG conditions; one text report per experiment.

Place beside database.py, schema.sql, prepare_data.py, retrieve.py,
rerank.py and generate.py. Run: python run_baseline.py
Uses data/docs/non_pois and data/docs/pois. Questions are assumed to
have been human-verified as clean-correct. Project tables are reset.
"""
import gc
import importlib
import os
import time
from datetime import datetime
from pathlib import Path

CONDITIONS = ("Clean", "Clean + poison", "Poison only")
FLAGS = {
    "retrieval": "Poison retrieved",
    "reranker_input": "Poison entered reranker",
    "context": "Poison selected in top 3",
    "generator": "Poison sent to generator",
    "promotion": "Poison promoted",
}
PAGE_FIELDS = (
    "filepath", "page_num", "ispoisoned", "score",
    "retrieval_rank", "reranker_score", "reranker_rank",
)


def yes_no(prompt):
    while True:
        value = input(prompt + " (y/n): ").strip().lower()
        if value in ("y", "yes", "n", "no"):
            return value in ("y", "yes")
        print("Please enter y or n.")


def release_model(module):
    import torch
    module.setup.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reset_database(database):
    """Drop the five project tables and recreate empty tables."""
    conn = database.create_connection()
    try:
        with conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    DROP TABLE IF EXISTS question_embeddings;
                    DROP TABLE IF EXISTS page_embeddings;
                    DROP TABLE IF EXISTS questions;
                    DROP TABLE IF EXISTS pages;
                    DROP TABLE IF EXISTS documents;
                """)
                cursor.execute(Path("schema.sql").read_text(encoding="utf-8"))
    finally:
        conn.close()
    print("Database cleaned; empty project tables recreated.", flush=True)


def load_documents(modules, condition):
    """Import each file and verify its page/vector counts."""
    import pymupdf
    prepare = modules["prepare_data"]
    folders = {
        "Clean": [("non_pois", False)],
        "Clean + poison": [("non_pois", False), ("pois", True)],
        "Poison only": [("pois", True)],
    }
    corpus = []
    names = set()
    for folder, poisoned in folders[condition]:
        root = Path("data/docs") / folder
        files = sorted(p for p in root.iterdir()
                       if p.is_file() and p.suffix.lower() == ".pdf")
        if not files:
            raise ValueError(f"No PDFs in {root}")
        for path in files:
            if path.name in names:
                raise ValueError(f"Duplicate filename: {path.name}")
            names.add(path.name)
            with pymupdf.open(path) as pdf:
                pages = len(pdf)
            if not pages:
                raise ValueError(f"Empty PDF: {path}")
            corpus.append((str(path), poisoned, pages))

    try:
        for path, poisoned, _ in corpus:
            print(f"Importing {path}...", flush=True)
            success, _ = prepare.insert_document(path, Path(path).name, poisoned)
            if not success:
                raise RuntimeError(f"Import failed: {path}")
    finally:
        release_model(prepare)

    conn = modules["database"].create_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT d.filepath, d.ispoisoned, COUNT(DISTINCT p.id),
                       COUNT(DISTINCT e.page_id), COUNT(e.page_id)
                FROM documents d
                LEFT JOIN pages p ON p.docid = d.id
                LEFT JOIN page_embeddings e ON e.page_id = p.id
                GROUP BY d.id, d.filepath, d.ispoisoned
                ORDER BY d.filepath
            """)
            actual = cursor.fetchall()
    finally:
        conn.close()
    expected = sorted(corpus)
    observed = sorted((p, flag, count) for p, flag, count, _, _ in actual)
    if observed != expected or any(count != embedded or not vectors
                                  for _, _, count, embedded, vectors in actual):
        raise RuntimeError("Indexed corpus or embedding counts are incomplete.")
    return actual


def summarize(pages):
    return [{key: p.get(key) for key in PAGE_FIELDS} for p in pages]


def run_question(question, modules, result):
    try:
        pages = modules["retrieve"].retrieve_pages(question, top_k=10)
    finally:
        release_model(modules["retrieve"])
    if not pages:
        raise RuntimeError("No pages retrieved")
    for rank, page in enumerate(pages, 1):
        page["retrieval_rank"] = rank
    result["retrieved"] = summarize(pages)
    result["retrieval"] = any(p["ispoisoned"] for p in pages)
    result["reranker_input"] = result["retrieval"]

    try:
        ranked = modules["rerank"].rerank_pages(question, pages, top_k=len(pages))
    finally:
        release_model(modules["rerank"])
    selected = ranked[:3]
    result["reranked"] = summarize(ranked)
    result["selected"] = summarize(selected)
    result["context"] = any(p["ispoisoned"] for p in selected)
    result["promotion"] = any(p["ispoisoned"] and
        p["reranker_rank"] < p["retrieval_rank"] for p in ranked)
    if not selected:
        raise RuntimeError("No context pages selected")

    try:
        result["answer"] = modules["generate"].generate_answer(question, selected)
    finally:
        release_model(modules["generate"])
    result["generator"] = result["context"]
    result["status"] = "completed"


def table(headers, rows):
    rows = [[str(x) for x in headers]] + [[str(x) for x in r] for r in rows]
    widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    return "\n".join(" | ".join(v.ljust(w) for v, w in zip(r, widths)) for r in rows)


def report_text(questions, runs):
    lines = ["VISUAL RAG BASELINE COMPARISON", "",
        "Settings: retrieval top 10; generator top 3; existing model prompts.",
        "Input questions are assumed human-verified as clean-correct.",
        "Changed text is not automatically wrong. No automatic answer grading.",
        "Exposure labels cover entire poisoned documents, including unedited pages.",
        "", "EXECUTION SUMMARY"]
    rows = []
    for name in CONDITIONS:
        run = runs.get(name, {})
        records = run.get("results", [])
        done = sum(r["status"] == "completed" for r in records)
        rows.append([name, run.get("status", "not run"), len(records), done,
                     sum(r["status"] == "failed" for r in records),
                     f"{run.get('seconds', 0):.1f}"])
    lines.append(table(["Condition", "Status", "Attempted", "Completed", "Failed", "Seconds"], rows))
    lines += ["", "ANSWER CHANGES AGAINST PAIRED COMPLETED RUNS"]
    rows = []
    for left, right in [("Clean", "Clean + poison"), ("Clean", "Poison only"),
                        ("Clean + poison", "Poison only")]:
        pairs = [(a, b) for a, b in zip(runs.get(left, {}).get("results", []),
                                      runs.get(right, {}).get("results", []))
                 if a["status"] == b["status"] == "completed"]
        changed = sum(" ".join(a["answer"].split()) != " ".join(b["answer"].split())
                      for a, b in pairs)
        rate = f"{100 * changed / len(pairs):.1f}%" if pairs else "N/A"
        rows.append([f"{left} vs {right}", changed, len(pairs), rate])
    lines.append(table(["Comparison", "Changed", "Paired", "Change rate"], rows))
    lines += ["", "POISON EXPOSURE AND PROMOTION"]
    rows = []
    for flag, label in FLAGS.items():
        row = [label]
        for name in CONDITIONS:
            values = [r[flag] for r in runs.get(name, {}).get("results", []) if flag in r]
            row.append(f"{sum(values)}/{len(values)} ({100 * sum(values)/len(values):.1f}%)"
                       if values else "Not measured")
        rows.append(row)
    lines.append(table(["Metric", *CONDITIONS], rows))
    lines += ["Rates use questions where that stage was measured; unknown stages are excluded.",
        "Generator exposure is measured after generation returns successfully.",
        "Reranker input exposure equals retrieval exposure in this pipeline.",
        "", "Human review: mark correctness and target matches separately."]
    for name, run in runs.items():
        lines += ["", f"{name.upper()} CORPUS"]
        for path, poisoned, pages, embedded, vectors in run.get("corpus", []):
            lines.append(f"{path} | poison={poisoned} | pages={pages} | vectors={vectors}")
        for error in run.get("errors", []):
            lines.append("RUN ERROR: " + error)
    for i, question in enumerate(questions):
        lines += ["", "=" * 70, f"Question {i + 1}: {question}"]
        for name in CONDITIONS:
            records = runs.get(name, {}).get("results", [])
            record = records[i] if i < len(records) else {}
            lines += ["", f"[{name}] {record.get('status', 'not run')}",
                      record.get("answer", record.get("error", "Not run"))]
            for stage in ("retrieved", "reranked", "selected"):
                if stage in record:
                    lines.append(f"  {stage}:")
                    for p in record[stage]:
                        lines.append(f"    {p['filepath']} page {p['page_num']} | "
                            f"poison={p['ispoisoned']} | retrieval={p['retrieval_rank']} "
                            f"score={p['score']} | reranker={p['reranker_rank']} "
                            f"score={p['reranker_score']}")
            lines.append("  Human review: correctness ____ target match ____ notes ____")
    return "\n".join(lines) + "\n"


def run_baselines(questions):
    # Lazy imports let the user choose Exit without a database connection.
    modules = {name: importlib.import_module(name) for name in
               ("database", "prepare_data", "retrieve", "rerank", "generate")}
    # Indexing and retrieval must use the same model settings on Colab.
    modules["prepare_data"].setup = modules["retrieve"].setup
    Path("reports").mkdir(exist_ok=True)
    path = Path("reports") / f"baselines_{datetime.now():%Y%m%d_%H%M%S_%f}.txt"
    runs = {}
    reset_database(modules["database"])
    try:
        for name in CONDITIONS:
            print(f"\n--- {name} ---", flush=True)
            run = {"status": "loading", "results": [], "errors": []}
            runs[name] = run
            started = time.perf_counter()
            try:
                run["corpus"] = load_documents(modules, name)
                run["status"] = "running"
                for i, question in enumerate(questions, 1):
                    print(f"\nQuestion {i}: {question}", flush=True)
                    result = {"status": "running"}
                    run["results"].append(result)
                    try:
                        run_question(question, modules, result)
                        print("Answer:", result["answer"], flush=True)
                    except Exception as error:
                        result.update(status="failed", error=str(error))
                        print("Failed:", error, flush=True)
                    # Cleanup again after exception traceback references are gone.
                    for module in ("retrieve", "rerank", "generate"):
                        release_model(modules[module])
                    path.write_text(report_text(questions, runs), encoding="utf-8")
                run["status"] = "completed"
                if any(r["status"] == "failed" for r in run["results"]):
                    run["status"] = "completed with failures"
            except (Exception, KeyboardInterrupt) as error:
                run["status"] = "stopped"
                run["errors"].append(str(error) or "Interrupted by user")
                for result in run["results"]:
                    if result["status"] == "running":
                        result.update(status="failed", error="Interrupted")
            finally:
                try:
                    reset_database(modules["database"])
                except Exception as error:
                    run["status"] = "stopped"
                    run["errors"].append(f"Database cleanup failed: {error}")
                run["seconds"] = time.perf_counter() - started
                path.write_text(report_text(questions, runs), encoding="utf-8")
            print(f"Report saved: {path.resolve()}")
            if run["status"] == "stopped":
                print("Experiment stopped:", "; ".join(run["errors"]))
                break
            if name != CONDITIONS[-1] and not yes_no("Continue to the next baseline?"):
                break
    finally:
        for name in ("prepare_data", "retrieve", "rerank", "generate"):
            release_model(modules[name])
        text = report_text(questions, runs)
        path.write_text(text, encoding="utf-8")
        print("\n" + text.split("Human review:")[0])
        print(f"Full report: {path.resolve()}")


def main():
    os.chdir(Path(__file__).resolve().parent)
    print("Uses PDFs in data/docs/non_pois and data/docs/pois.")
    print("WARNING: the five project tables and their data will be reset.")
    print("PDF files will not be deleted. Back up database data you need.")
    while True:
        print("\n1. Ask one question\n2. Load questions from a text file\n3. Exit")
        choice = input("Choose an option: ").strip()
        if choice == "3":
            break
        try:
            if choice == "1":
                questions = [input("Question: ").strip()]
            elif choice == "2":
                path = Path(input("Question file path: ").strip().strip("\"'")).expanduser()
                questions = [s.strip() for s in path.read_text(encoding="utf-8-sig").splitlines()
                             if s.strip()]
            else:
                print("Please choose 1, 2 or 3.")
                continue
            if not questions or not all(questions):
                print("Provide at least one non-empty question.")
                continue
            run_baselines(questions)
        except (Exception, KeyboardInterrupt) as error:
            print("Stopped:", str(error) or "Interrupted by user")
        if not yes_no("Return to the question menu?"):
            break


if __name__ == "__main__":
    main()
