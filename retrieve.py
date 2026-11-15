from functools import lru_cache
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from pgvector.psycopg2 import register_vector

import database



_settings = {}


def adapter_key(name):
    """Align the old/new backbone names when checking the loaded tensors."""
    while name.startswith(("base_model.model.", "model.", "language_model.")):
        for prefix in ("base_model.model.", "model.", "language_model."):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
    return re.sub(r"(\.lora_[AB])\.[^.]+\.weight$", r"\1.weight", name)


def verify_adapter_weights(model, checkpoint_path):
    """Stop if any trained LoRA tensor was dropped or left at its initial value."""
    from safetensors import safe_open
    loaded = {}
    for name, tensor in model.named_parameters():
        if ".lora_A." not in name and ".lora_B." not in name:
            continue
        key = adapter_key(name)
        if key in loaded:
            raise RuntimeError(f"Ambiguous retriever adapter tensor: {key}")
        loaded[key] = tensor
    missing, different, total = [], [], 0
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
        for name in checkpoint.keys():
            if ".lora_A." not in name and ".lora_B." not in name:
                continue
            total += 1
            key = adapter_key(name)
            if key not in loaded:
                missing.append(name)
                continue
            actual = loaded[key].detach().cpu()
            expected = checkpoint.get_tensor(name).to(dtype=actual.dtype)
            if actual.shape != expected.shape or not torch.equal(actual, expected):
                different.append(name)
    if not total or missing or different:
        examples = (missing + different)[:3]
        raise RuntimeError(
            f"Retriever adapter verification failed: {len(missing)} missing, "
            f"{len(different)} mismatched, {total} expected LoRA tensors. "
            f"Examples: {examples}. Stopped before indexing; do not use this model's results.")
    active = model.active_adapters()
    if not active:
        raise RuntimeError("Retriever LoRA tensors are loaded but no adapter is active.")
    print(f"Verified {total}/{total} trained retriever LoRA tensors; active adapters: {active}", flush=True)


def configure(settings):
    global _settings
    if settings != _settings:
        setup.cache_clear()
    _settings = dict(settings)


@lru_cache(maxsize=1)
def setup():
    spec = _settings.get("models", {}).get("retriever", {})
    model_name = spec.get("id", "vidore/colqwen2-v1.0")
    default_base = {"id": "vidore/colqwen2-base"} if model_name == "vidore/colqwen2-v1.0" else None
    base = _settings.get("models", {}).get("retriever_base", default_base)

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    # Use ColQwen2's automatic adapter-loading path, as in the working v1.
    # Cache the pinned adapter locally first. Its commit must not be used as
    # the revision of the separate base-model repository.
    adapter_path = None
    checkpoint = model_name
    revision = spec.get("revision")
    if base:
        from huggingface_hub import snapshot_download
        adapter_path = Path(snapshot_download(
            repo_id=model_name, revision=spec.get("revision"),
            allow_patterns=["adapter_config.json", "adapter_model.safetensors"]))
        adapter_config = json.loads((adapter_path / "adapter_config.json").read_text())
        if adapter_config["base_model_name_or_path"] != base["id"]:
            raise ValueError("Retriever base setting does not match the adapter checkpoint")
        checkpoint = str(adapter_path)
        revision = base.get("revision")
    model = ColQwen2.from_pretrained(
        checkpoint,
        revision=revision,
        torch_dtype=dtype,
        device_map=device,
    ).eval()
    if adapter_path:
        verify_adapter_weights(model, adapter_path / "adapter_model.safetensors")

    processor = ColQwen2Processor.from_pretrained(model_name, revision=spec.get("revision"),
                                                  max_num_visual_tokens=_settings.get("max_visual_tokens", 1024))

    return model, processor


def retrieve_pages(question, top_k=10, allowed_page_ids=None, page_metadata=None,
                   strict_top_k=False, tie_seed=0):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    if allowed_page_ids is not None and not allowed_page_ids:
        return []

    model, processor = setup()
    inputs = processor.process_queries([question]).to(model.device)

    with torch.inference_mode():
        question_embeddings = model(**inputs)[0].float().cpu()

    conn = database.create_connection()

    try:
        register_vector(conn)

        with conn.cursor() as curr:
            query = """
                SELECT
                    p.id, p.docid, p.page_num, p.text,
                    p.ispoisoned, d.filepath, e.embedding
                FROM pages p
                JOIN documents d ON d.id = p.docid
                JOIN page_embeddings e ON e.page_id = p.id
            """
            params = ()
            if allowed_page_ids is not None:
                query += " WHERE p.id = ANY(%s)"
                params = (list(allowed_page_ids),)
            query += " ORDER BY p.id, e.vector_position"
            curr.execute(query, params)

            pages = {}

            for row in curr:
                page_id, docid, page_num, text, poisoned, path, vector = row

                if page_id not in pages:
                    pages[page_id] = {
                        "page_id": page_id,
                        "document_id": docid,
                        "page_num": page_num,
                        "text": text,
                        "ispoisoned": poisoned,
                        "filepath": path,
                        "vectors": [],
                    }

                # Handle both pgvector Vector objects and NumPy arrays.
                if hasattr(vector, "to_numpy"):
                    vector = vector.to_numpy()

                pages[page_id]["vectors"].append(
                    np.asarray(vector, dtype=np.float32)
                )

    finally:
        conn.close()

    results = []
    page_list = list(pages.values())

    for start in range(0, len(page_list), 8):
        batch = page_list[start:start + 8]

        embeddings = [
            torch.from_numpy(np.stack(page["vectors"]))
            for page in batch
        ]

        with torch.inference_mode():
            scores = processor.score_multi_vector(
                [question_embeddings],
                embeddings
            )[0]

        for page, score in zip(batch, scores):
            page.pop("vectors")
            page["score"] = score.item()
            if page_metadata is not None:
                db_id = page["page_id"]
                page.update(page_metadata[db_id])
                page["database_page_id"] = db_id
            results.append(page)

    if strict_top_k:
        results.sort(key=lambda p: (-p["score"], hashlib.sha256(
            f"{tie_seed}:{p['page_id']}".encode()).hexdigest()))
    else:
        results.sort(key=lambda page: page["score"], reverse=True)

    if not results:
        return []

    # Include every page tied with the kth page.
    cutoff = results[min(top_k, len(results)) - 1]["score"]

    selected = results[:top_k] if strict_top_k else [
        page for page in results
        if page["score"] >= cutoff
    ]

    for rank, page in enumerate(selected, start=1):
        page["retrieval_rank"] = rank

    return selected


if __name__ == "__main__":
    # A loading check only: no SQL indexing, retrieval, or answer generation.
    import argparse
    from load_dataset import read_config
    from run_experiment_a import Backend
    parser = argparse.ArgumentParser(description="Load and verify the retrieval model only")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    backend = Backend(read_config(args.config))
    backend.freeze_revisions()
    configure(backend.runtime_cfg)
    setup()
    print("Retriever loading check passed.")
