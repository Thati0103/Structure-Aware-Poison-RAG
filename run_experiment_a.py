"""Run Experiment A, or both experiments with --both.

main() prepares the jobs. execute() runs retrieval, reranking, then generation.
Backend calls the model/database modules. RunProgress handles saving and reuse.
"""
import argparse
import gc
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from load_dataset import (ATTACKS, clean_corpus, digest, file_hash, make_a_jobs,
                          make_b_jobs, read_config, prepare_dataset)

from prompts import GENERATOR_PROMPT, NO_CONTEXT_PROMPT, RERANK_PROMPT


def environment_info():
    versions = {}
    for name in ("torch", "transformers", "colpali-engine", "peft", "accelerate", "numpy", "PyMuPDF", "huggingface-hub"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


class Backend:
    """Orchestrate the existing pipeline modules; embeddings remain in PostgreSQL."""

    def __init__(self, cfg):
        import torch
        self.cfg = cfg
        self.torch = torch
        self.modules = {}
        self.observed = {}
        self.page_ids = {}
        torch.manual_seed(cfg["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg["seed"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            self.inference_dtype = "float32"
        elif torch.cuda.is_bf16_supported():
            self.inference_dtype = "bfloat16"
        else:
            self.inference_dtype = "float16"
        self.hardware = torch.cuda.get_device_name(0) if self.device == "cuda" else "cpu"
        self.state = Path(cfg["state_dir"])
        self.state.mkdir(parents=True, exist_ok=True)

    def freeze_revisions(self):
        from huggingface_hub import HfApi
        specs = self.cfg["models"]
        # An adapter commit belongs to its own repository. Lock the base
        # separately so its weights are never requested at an adapter commit.
        adapter_bases = {
            "retriever": ("vidore/colqwen2-v1.0", "vidore/colqwen2-base"),
            "reranker": ("lightonai/MonoQwen2-VL-v0.1", "Qwen/Qwen2-VL-2B-Instruct"),
        }
        for stage, (adapter_id, base_id) in adapter_bases.items():
            if specs.get(stage, {}).get("id") == adapter_id:
                specs.setdefault(f"{stage}_base", {"id": base_id, "revision": None})
        requested = {
            stage: {"id": spec["id"], "revision": spec.get("revision") or "main"}
            for stage, spec in specs.items()
        }
        path = self.state / f"model_revisions_{digest(requested)[:16]}.json"
        if path.exists():
            lock = json.loads(path.read_text())
            if lock["requested"] != requested:
                raise ValueError("Saved model-revision record does not match its settings")
        else:
            api = HfApi()
            resolved = {
                stage: api.model_info(spec["id"], revision=spec["revision"]).sha
                for stage, spec in requested.items()
            }
            lock = {"requested": requested, "resolved": resolved}
            save_json(path, lock)
        self.runtime_cfg = json.loads(json.dumps(self.cfg))
        for stage, revision in lock["resolved"].items():
            self.runtime_cfg["models"][stage]["revision"] = revision
        self.signature = digest({
            "models": lock,
            "dpi": self.cfg["dpi"],
            "max_visual_tokens": self.cfg["max_visual_tokens"],
            "encoder_code": {
                name: file_hash(Path(__file__).parent / name)
                for name in ("prepare_data.py", "retrieve.py")
            },
            "versions": environment_info(),
            "device": self.device,
            "hardware": getattr(self, "hardware", self.device),
            "bf16": self.torch.cuda.is_bf16_supported() if self.device == "cuda" else False,
        })
        return lock

    def module(self, name):
        import importlib
        if name not in self.modules:
            module = importlib.import_module(name)
            self.modules[name] = module
            if name == "database":
                settings = dict(self.cfg["database"])
                schema = settings.get("schema", "")
                if not schema or schema == "public" or schema.startswith("pg_"):
                    raise ValueError("Experiment indexing requires a separate database schema")
                settings["schema"] = schema[:48] + "_" + self.signature[:12]
                self.database_schema = settings["schema"]
                module.configure(settings)
            else:
                module.configure(self.runtime_cfg)
        return self.modules[name]

    def observe(self, module, stage):
        model, _ = module.setup()
        observed = {"base_model": getattr(model.config, "_name_or_path", None),
                    "base_commit": getattr(model.config, "_commit_hash", None),
                    "dtype": str(next(model.parameters()).dtype)}
        path = self.state / f"{self.signature}_{stage}_observed.json"
        if path.exists() and json.loads(path.read_text()) != observed:
            raise ValueError(f"Observed {stage} base checkpoint changed; use a new state_dir and schema")
        save_json(path, observed)
        self.observed[stage] = observed

    def release(self):
        for name, module in self.modules.items():
            if name != "database":
                module.setup.cache_clear()
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def index(self, pages, rows):
        import pymupdf

        db = self.module("database")
        db.create_schema()
        files = {}
        for page in pages:
            path = page["filepath"]
            files[path] = files.get(path, False) or page["ispoisoned"]
        for position, (path, poisoned) in enumerate(files.items(), start=1):
            content_hash = file_hash(path)
            with pymupdf.open(path) as pdf:
                expected_pages = len(pdf)
            if self.document_is_indexed(db, path, content_hash, expected_pages):
                continue
            print(f"Indexing {position}/{len(files)}: {Path(path).name}", flush=True)
            self.index_document(db, path, poisoned, content_hash)

        connection = db.create_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute("""SELECT d.filepath, p.page_num, p.id FROM documents d
                    JOIN pages p ON p.docid = d.id WHERE d.filepath = ANY(%s)""", (list(files),))
                self.page_ids = {(path, number): page_id for path, number, page_id in cursor}
        finally:
            connection.close()
        if any((page["filepath"], page["page_num"]) not in self.page_ids for page in pages):
            raise RuntimeError("Indexed page mapping is incomplete")

    def document_is_indexed(self, db, path, content_hash, expected_pages):
        connection = db.create_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT id, content_hash, embedding_signature FROM documents WHERE filepath = %s",
                    (path,),
                )
                record = cursor.fetchone()
                if not record or record[1:] != (content_hash, self.signature):
                    return False
                cursor.execute("""SELECT COUNT(DISTINCT p.id), COUNT(DISTINCT e.page_id)
                    FROM pages p LEFT JOIN page_embeddings e ON e.page_id = p.id
                    WHERE p.docid = %s""", (record[0],))
                return cursor.fetchone() == (expected_pages, expected_pages)
        finally:
            connection.close()

    def index_document(self, db, path, poisoned, content_hash):
        retriever = self.module("retrieve")
        prepare = self.module("prepare_data")
        self.observe(retriever, "retriever")
        success, document_id = prepare.insert_document(path, Path(path).name, poisoned)
        if not success:
            raise RuntimeError(f"Failed to index {path}")
        connection = db.create_connection()
        try:
            with connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "UPDATE documents SET content_hash = %s, embedding_signature = %s WHERE id = %s",
                        (content_hash, self.signature, document_id),
                    )
        finally:
            connection.close()

    def retrieve(self, question, pages, qid):
        module = self.module("retrieve")
        self.observe(module, "retriever")
        metadata = {self.page_ids[(p["filepath"], p["page_num"])]: p for p in pages}
        return module.retrieve_pages(question, top_k=self.cfg["retrieval_k"],
                    allowed_page_ids=list(metadata), page_metadata=metadata, strict_top_k=True,
                    tie_seed=f"{self.cfg['seed']}:{qid}")

    def rerank(self, question, pages):
        module = self.module("rerank")
        self.observe(module, "reranker")
        return module.rerank_pages(question, pages, top_k=len(pages))

    def generate(self, question, pages):
        module = self.module("generate")
        self.observe(module, "generator")
        return module.generate_answer(question, pages, max_new_tokens=self.cfg["max_new_tokens"],
                                      allow_no_context=not pages)


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def selected_attacks(experiment, conditions, requested):
    if not conditions:
        return requested
    return [attack for attack in requested if any(attack in condition for condition in conditions)]


def corpus_check(cfg, jobs, allow_small=False):
    if jobs[0]["experiment"] != "A":
        return None
    clean_count = sum(page["slot"] == "clean" for page in jobs[0]["pages"])
    retrieval_k = cfg["retrieval_k"]
    suggested = cfg.get("corpus_warning_multiplier", 5) * retrieval_k
    if clean_count <= retrieval_k and not (allow_small or cfg.get("allow_small_corpus", False)):
        raise ValueError(f"Clean corpus has {clean_count} pages but retrieval_k={retrieval_k}. "
                         f"Add distractor PDFs to clean/ (recommended at least {suggested} pages). "
                         "Use --clean for a smoke test, or --allow-small-corpus only for debugging.")
    info = {"clean_pages": clean_count, "retrieval_k": retrieval_k, "recommended_clean_pages": suggested,
            "candidate_pages_min": min(len(job["pages"]) for job in jobs),
            "candidate_pages_max": max(len(job["pages"]) for job in jobs),
            "smoke_test_only": clean_count <= retrieval_k, "below_recommended_size": clean_count < suggested}
    if clean_count < suggested:
        print(f"Corpus check: {clean_count} clean pages, retrieval top {retrieval_k}; "
              f"recommend at least {suggested} clean pages with distractors."
              + (" Smoke test only: top-k can include the entire corpus." if clean_count <= retrieval_k else ""), flush=True)
    return info


def prepare_jobs(cfg, experiment, attacks, conditions, verify_edits, allow_small_corpus=False):
    rows = prepare_dataset(cfg, attacks, verify_edits=verify_edits)
    if experiment == "A":
        jobs = make_a_jobs(rows, clean_corpus(cfg), attacks, conditions)
    else:
        jobs = make_b_jobs(rows, attacks, conditions)
    if not jobs:
        raise ValueError("No experiment conditions selected")
    if conditions:
        missing = set(conditions) - {job["condition"] for job in jobs}
        if missing:
            raise ValueError(f"Unknown conditions or attacks not enabled: {sorted(missing)}")
    check = corpus_check(cfg, jobs, allow_small_corpus)
    for job in jobs:
        job["corpus_check"] = check
    return rows, jobs


def choose_run_folder(results_dir, experiment, fingerprint, label=None):
    """Same inputs resume; changed inputs get a separate folder automatically."""
    if label is not None and (not label or Path(label).name != label or label in (".", "..")):
        raise ValueError("run-name must be one folder name")
    prefix = label or f"experiment_{experiment.lower()}"
    for length in (12, 20, 32, 64):
        folder = Path(results_dir) / f"{prefix}_{fingerprint[:length]}"
        manifest_path = folder / "manifest.json"
        if not folder.exists():
            return folder
        if manifest_path.exists():
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            if saved["fingerprint"] == fingerprint:
                return folder
    raise ValueError("Could not choose a run folder; inspect conflicting result folders")


CACHE_FIELDS = ("status", "error", "retrieved", "reranked", "selected", "prediction")


def backend_runtime(backend):
    device = getattr(backend, "device", "test")
    return {"device": device, "dtype": getattr(backend, "inference_dtype", "unknown"),
            "hardware": getattr(backend, "hardware", device)}


def case_fingerprint(cfg, job, hashes, lock, backend, code_hashes):
    """Only inputs that affect this question/condition's model output."""
    is_full_pipeline = job["experiment"] == "A"
    if is_full_pipeline:
        code_names = (
            "run_experiment_a.py", "prompts.py", "database.py", "prepare_data.py",
            "retrieve.py", "rerank.py", "generate.py",
        )
        model_identity = {"lock": lock, "backend": backend.signature}
    else:
        code_names = ("run_experiment_a.py", "run_experiment_b.py", "prompts.py", "generate.py")
        model_identity = {
            section: {"generator": values["generator"]}
            for section, values in lock.items()
            if section in ("requested", "resolved") and "generator" in values
        }
        if not model_identity:  # Test doubles have a deliberately simple lock.
            model_identity = {"backend": backend.signature, "lock": lock}
    settings = {name: cfg[name] for name in ("dpi", "seed", "max_new_tokens")}
    if is_full_pipeline:
        for name in ("retrieval_k", "context_pages", "max_visual_tokens"):
            settings[name] = cfg[name]
    pages = [
        {"page": page, "pdf_hash": hashes[page["filepath"]]} for page in job["pages"]
    ]
    return digest({
        "cache_version": 1,
        "experiment": job["experiment"],
        "condition": job["condition"],
        "question_id": job["row"]["question_id"],
        "question": job["row"]["question"],
        "pages": pages,
        "settings": settings,
        "models": model_identity,
        "code": {name: code_hashes[name] for name in code_names if name in code_hashes},
        "environment": environment_info(),
        "runtime": backend_runtime(backend),
        "prompt": GENERATOR_PROMPT if job["pages"] else NO_CONTEXT_PROMPT,
        "reranker_prompt": RERANK_PROMPT if is_full_pipeline else None,
    })


class RunProgress:
    """Keep one run's records, checkpoints, and reusable model outputs together."""

    def __init__(self, cfg, rows, jobs, backend, run_name):
        self.cfg = cfg
        self.backend = backend
        self.experiment = jobs[0]["experiment"]
        pdf_paths = sorted({page["filepath"] for job in jobs for page in job["pages"]})
        pdf_hashes = {path: file_hash(path) for path in pdf_paths}
        model_lock = backend.freeze_revisions()
        code_hashes = {
            path.name: file_hash(path)
            for path in Path(__file__).parent.glob("*.py")
            if path.name != "evaluate.py"
        }
        inputs = {
            "config": cfg,
            "rows": rows,
            "pdf_hashes": pdf_hashes,
            "conditions": sorted({job["condition"] for job in jobs}),
            "experiment": self.experiment,
            "model_lock": model_lock,
            "backend_signature": backend.signature,
            "code_hashes": code_hashes,
            "runtime": backend_runtime(backend),
            "generator_prompt": GENERATOR_PROMPT,
            "no_context_prompt": NO_CONTEXT_PROMPT,
            "reranker_prompt": RERANK_PROMPT,
        }
        self.fingerprint = digest(inputs)
        self.folder = choose_run_folder(
            cfg["results_dir"], self.experiment, self.fingerprint, run_name
        )
        self.folder.mkdir(parents=True, exist_ok=True)
        self.manifest = self.load_manifest(inputs, jobs[0].get("corpus_check"))

        records_path = self.folder / "records.json"
        previous_records = json.loads(records_path.read_text()) if records_path.exists() else []
        self.records = {record["record_key"]: record for record in previous_records}
        self.cache_dir = Path(cfg["state_dir"]) / "cases"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.saved_cache_hashes = {}

        for job in jobs:
            question_id = job["row"]["question_id"]
            job["key"] = digest([self.experiment, question_id, job["condition"]])[:24]
            job["case_key"] = case_fingerprint(
                cfg, job, pdf_hashes, model_lock, backend, code_hashes
            )
            if job["key"] not in self.records:
                record = self.new_record(job)
                self.restore_cached_output(record)
                self.records[job["key"]] = record

            record = self.record_for(job)
            record["tracked_pages"] = job.get("tracked_pages", {})
            record["case_fingerprint"] = job["case_key"]

    def load_manifest(self, inputs, corpus_check):
        manifest_path = self.folder / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["fingerprint"] != self.fingerprint:
                raise ValueError("Run manifest changed while opening results; retry the command")
            return manifest

        manifest = {
            "fingerprint": self.fingerprint,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "packages": environment_info(),
            "inputs": inputs,
            "setting": "Localized, per-question injection; clean originals retained",
            "corpus_check": corpus_check,
            "insertion": "One inserted document per question (only the target page edited)",
            "clean_corpus": "One shared clean corpus for every question and condition",
            "tie_policy": "Strict top-k; equal scores ordered by seeded neutral page-ID hash",
        }
        save_json(manifest_path, manifest)
        return manifest

    def new_record(self, job):
        row = job["row"]
        attack = job["attack"]
        is_full_pipeline = self.experiment == "A"
        return {
            "record_key": job["key"],
            "run_fingerprint": self.fingerprint,
            "experiment": self.experiment,
            "condition": job["condition"],
            "question_id": row["question_id"],
            "question": row["question"],
            "doc_name": row["doc_name"],
            "support_pages": row["support"],
            "clean_answer": row["clean_answer"],
            "target_answer": row.get(f"{attack}_target_answer", "") if attack else "",
            "attack": attack,
            "baseline": job["baseline"],
            "tracked_pages": job.get("tracked_pages", {}),
            "retrieval_k": self.cfg["retrieval_k"],
            "context_pages_limit": self.cfg["context_pages"] if is_full_pipeline else None,
            "answer_format": "final_answer_line_v1",
            "corpus_pages": len(job["pages"]) if is_full_pipeline else None,
            "case_fingerprint": job["case_key"],
            "status": "pending",
        }

    def restore_cached_output(self, record):
        case_key = record["case_fingerprint"]
        cache_path = self.cache_dir / f"{case_key}.json"
        if not cache_path.exists():
            return
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("case_fingerprint") != case_key:
            return
        if cached.get("status") == "completed" and "prediction" not in cached:
            return

        record.update({field: cached[field] for field in CACHE_FIELDS if field in cached})
        record["reused_from_cache"] = True
        record["cache_source_run"] = cached.get("source_run")
        self.manifest.setdefault("observed_models", {}).update(cached.get("observed_models", {}))
        self.saved_cache_hashes[case_key] = digest(cached)

    def record_for(self, job):
        return self.records[job["key"]]

    def save(self):
        for record in self.records.values():
            if "selected" in record:
                record["context_pages_supplied"] = len(record["selected"])
        save_json(self.folder / "records.json", list(self.records.values()))
        self.save_cached_outputs()

        self.manifest.setdefault("observed_models", {}).update(self.backend.observed)
        if getattr(self.backend, "database_schema", None):
            self.manifest["database_schema"] = self.backend.database_schema
        save_json(self.folder / "manifest.json", self.manifest)

        completed = sum(record["status"] == "completed" for record in self.records.values())
        latest_path = Path(self.cfg["results_dir"]) / f"latest_{self.experiment.lower()}.json"
        save_json(latest_path, {
            "experiment": self.experiment,
            "run_dir": self.folder.name,
            "fingerprint": self.fingerprint,
            "completed": completed,
            "total": len(self.records),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        })

    def save_cached_outputs(self):
        for record in self.records.values():
            if record["status"] == "pending":
                continue
            payload = {field: record[field] for field in CACHE_FIELDS if field in record}
            payload.update(
                case_fingerprint=record["case_fingerprint"],
                source_run=(record.get("inference_source_run")
                            or record.get("cache_source_run") or str(self.folder)),
                observed_models={**self.manifest.get("observed_models", {}), **self.backend.observed},
            )
            case_key = record["case_fingerprint"]
            content_hash = digest(payload)
            if self.saved_cache_hashes.get(case_key) != content_hash:
                save_json(self.cache_dir / f"{case_key}.json", payload)
                self.saved_cache_hashes[case_key] = content_hash


def retrieve_pending_jobs(pending_jobs, rows, backend, progress):
    needing_retrieval = [
        job for job in pending_jobs if not progress.record_for(job).get("retrieved")
    ]
    if needing_retrieval:
        pages_to_index = [page for job in needing_retrieval for page in job["pages"]]
        backend.index(pages_to_index, rows)

    for job in needing_retrieval:
        record = progress.record_for(job)
        try:
            print(f"Retrieving {record['question_id']} {record['condition']}", flush=True)
            record["inference_source_run"] = str(progress.folder)
            record["retrieved"] = backend.retrieve(
                record["question"], job["pages"], record["question_id"]
            )
            if not record["retrieved"]:
                raise RuntimeError("No retrieved pages")
            record.update(status="retrieved", error=None)
        except Exception as exc:
            record.update(status="failed", error=f"retrieval: {type(exc).__name__}: {exc}")
        progress.save()
    backend.release()


def rerank_pending_jobs(pending_jobs, backend, progress):
    for job in pending_jobs:
        record = progress.record_for(job)
        if record.get("selected") or not record.get("retrieved"):
            continue
        try:
            print(f"Reranking {record['question_id']} {record['condition']}", flush=True)
            record["inference_source_run"] = str(progress.folder)
            record["reranked"] = backend.rerank(record["question"], record["retrieved"])
            selected_pages = record["reranked"][:progress.cfg["context_pages"]]
            if not selected_pages:
                raise RuntimeError("No selected context pages")
            record["selected"] = selected_pages
            record.update(status="context_ready", error=None)
        except Exception as exc:
            record.update(status="failed", error=f"reranking: {type(exc).__name__}: {exc}")
            backend.release()
        progress.save()
    backend.release()


def generate_pending_jobs(pending_jobs, backend, progress):
    for job in pending_jobs:
        record = progress.record_for(job)
        if "selected" not in record:
            continue
        if record["experiment"] == "A" and not record["selected"]:
            continue
        try:
            print(f"Generating {record['question_id']} {record['condition']}", flush=True)
            record["inference_source_run"] = str(progress.folder)
            record["prediction"] = backend.generate(record["question"], record["selected"])
            record.update(status="completed", error=None)
        except Exception as exc:
            record.update(status="failed", error=f"generation: {type(exc).__name__}: {exc}")
            backend.release()
        progress.save()


def execute(cfg, rows, jobs, backend, run_name=None):
    """Resume a run, process each model stage, then export results for review."""
    from evaluate import evaluate_run

    progress = RunProgress(cfg, rows, jobs, backend, run_name)
    progress.save()
    pending_jobs = [
        job for job in jobs if progress.record_for(job)["status"] != "completed"
    ]
    print(f"Results: {progress.folder}", flush=True)
    print(
        f"Reusing {len(jobs) - len(pending_jobs)} completed question/condition results; "
        f"{len(pending_jobs)} need work.", flush=True,
    )
    try:
        if progress.experiment == "A" and pending_jobs:
            retrieve_pending_jobs(pending_jobs, rows, backend, progress)
            rerank_pending_jobs(pending_jobs, backend, progress)
        elif progress.experiment == "B":
            for job in pending_jobs:
                progress.record_for(job).update(selected=job["pages"], status="context_ready")
            progress.save()
        generate_pending_jobs(pending_jobs, backend, progress)
    finally:
        backend.release()
        progress.save()

    review_path = progress.folder / "reviewed_answers.csv"
    evaluate_run(progress.folder, review_path if review_path.exists() else None)
    records = list(progress.records.values())
    incomplete = sum(record["status"] != "completed" for record in records)
    print(f"Saved {len(records)} records ({incomplete} incomplete) in {progress.folder}", flush=True)
    return progress.folder, records


def parse_arguments(experiment):
    parser = argparse.ArgumentParser(description=f"Experiment {experiment}: document poisoning pilot")
    parser.add_argument("--config", default=str(Path(__file__).resolve().with_name("config.json")))
    parser.add_argument(
        "--attacks", nargs="*", choices=ATTACKS, default=None,
        help="Override attacks selected in config.json",
    )
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument(
        "--run-name", default=None,
        help="Optional label; a settings identifier is added automatically",
    )
    parser.add_argument("--both", action="store_true", help="Run experiments A and B in sequence")
    parser.add_argument(
        "--clean", action="store_true",
        help="Run only the clean/duplication pilot; no edited PDFs needed",
    )
    parser.add_argument(
        "--allow-small-corpus", action="store_true",
        help="Allow a corpus no larger than top-k for debugging only",
    )
    parser.add_argument(
        "--skip-visual-check", action="store_true",
        help="Skip checking which PDF pages visibly changed",
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Validate inputs without loading models",
    )
    args = parser.parse_args()
    if args.conditions and (args.both or args.clean):
        parser.error("Use --conditions for one experiment, or use --both/--clean without --conditions")
    return args, parser


def main(experiment):
    args, parser = parse_arguments(experiment)
    cfg = read_config(args.config)
    requested = args.attacks if args.attacks is not None else cfg.get("attacks", ["cell"])
    if (
        not isinstance(requested, list)
        or any(attack not in ATTACKS for attack in requested)
        or len(requested) != len(set(requested))
    ):
        parser.error("config attacks must be a list drawn from cell, header, chart, layout, without duplicates")

    experiments = ("A", "B") if args.both else (experiment,)
    for current in experiments:
        conditions = args.conditions
        if args.clean:
            conditions = ["a0", "a1"] if current == "A" else ["b0", "b1"]
        attacks = selected_attacks(current, conditions, requested)
        rows, jobs = prepare_jobs(
            cfg, current, attacks, conditions, not args.skip_visual_check,
            allow_small_corpus=args.clean or args.allow_small_corpus,
        )
        condition_names = sorted({job["condition"] for job in jobs})
        print(
            f"Experiment {current}: validated {len(rows)} questions; {len(jobs)} runs; "
            f"conditions: {condition_names}", flush=True,
        )
        if not args.validate_only:
            backend = Backend(cfg)
            execute(cfg, rows, jobs, backend, args.run_name)


if __name__ == "__main__":
    main("A")
