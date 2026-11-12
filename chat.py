"""Interactive visual RAG: one question or a text file, with exposure reports.
Keep beside database.py, retrieve.py, rerank.py and generate.py.
Question files contain one question per nonempty line.
"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

FLAGS = (
    "poison_in_retrieval", "poison_entered_reranker",
    "poison_in_reranker_top_k", "poison_sent_to_generator",
    "poison_promoted",
)


def summarize(pages):
    fields = (
        "page_id", "document_id", "filepath", "page_num", "ispoisoned",
        "score", "retrieval_rank", "reranker_score", "reranker_rank",
    )
    return [{key: page.get(key) for key in fields} for page in pages]


def answer_question(question, retrieve_k=10, generate_k=3):
    from retrieve import retrieve_pages
    from rerank import rerank_pages
    from generate import generate_answer

    retrieved = retrieve_pages(
        question, top_k=retrieve_k,
    )
    for rank, page in enumerate(retrieved, start=1):
        page["retrieval_rank"] = rank

    reranked = rerank_pages(question, retrieved, top_k=len(retrieved)) if retrieved else []
    selected = reranked[:generate_k]
    answer = generate_answer(question, selected)
    return {
        "answer": answer,
        "poison_in_retrieval": any(p["ispoisoned"] for p in retrieved),
        "poison_entered_reranker": any(p["ispoisoned"] for p in retrieved),
        "poison_in_reranker_top_k": any(p["ispoisoned"] for p in selected),
        "poison_sent_to_generator": any(p["ispoisoned"] for p in selected),
        "poison_promoted": any(
            p["ispoisoned"] and p["reranker_rank"] < p["retrieval_rank"]
            for p in reranked
        ),
        "retrieved_pages": summarize(retrieved),
        "reranked_pages": summarize(reranked),
        "generator_pages": summarize(selected),
        "answer_manipulation": "not_evaluated",
    }


def run_questions(questions):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    folder = Path("reports") / run_id
    folder.mkdir(parents=True)
    records = []

    # Save each result immediately, so earlier results survive a later failure.
    with (folder / "results.jsonl").open("w", encoding="utf-8") as file:
        for number, question in enumerate(questions, start=1):
            print(f"\nQuestion {number}: {question}")
            record = {"question_id": number, "question": question}
            try:
                record.update(answer_question(question))
                record["status"] = "success"
                print("Answer:", record["answer"])
                for flag in FLAGS:
                    print(f"  {flag}: {record[flag]}")
            except Exception as error:
                record.update(status="error", error=f"{type(error).__name__}: {error}")
                print("Failed:", record["error"])

            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            records.append(record)

    columns = ["question_id", "question", "status", "answer", *FLAGS, "error"]
    with (folder / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    successful = [r for r in records if r["status"] == "success"]
    summary = {
        "run_id": run_id,
        "retrieve_k": 10,
        "generate_k": 3,
        "attempted": len(records),
        "successful": len(successful),
        "failed": len(records) - len(successful),
        "metrics": {
            flag: {
                "count": sum(r[flag] for r in successful),
                "rate": sum(r[flag] for r in successful) / len(successful) if successful else None,
            }
            for flag in FLAGS
        },
        "note": "Exposure tracking only; answer manipulation is not evaluated. Rates exclude failures.",
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nReports saved to: {folder}")


def main():
    while True:
        print("\n1. Ask one question\n2. Load questions from a text file\n3. Exit")
        choice = input("Choose an option: ").strip()

        if choice == "3":
            break

        try:
            if choice == "1":
                question = input("Question: ").strip()
                if not question:
                    print("Question cannot be empty.")
                    continue
                questions = [question]

            elif choice == "2":
                path = Path(input("Question file path: ").strip()).expanduser()
                questions = [
                    line.strip()
                    for line in path.read_text(encoding="utf-8-sig").splitlines()
                    if line.strip()
                ]
                if not questions:
                    print("The file contains no questions.")
                    continue

            else:
                print("Please choose 1, 2 or 3.")
                continue

            run_questions(questions)

        except Exception as error:
            print(f"Operation failed: {error}")


if __name__ == "__main__":
    main()
