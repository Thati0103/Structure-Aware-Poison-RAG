from functools import lru_cache

import numpy as np
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from pgvector.psycopg2 import register_vector

import database


@lru_cache(maxsize=1)
def setup():
    model_name = "vidore/colqwen2-v1.0"

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    model = ColQwen2.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    ).eval()

    processor = ColQwen2Processor.from_pretrained(model_name, max_num_visual_tokens=1024)

    return model, processor


def retrieve_pages(question, top_k=10):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    model, processor = setup()
    inputs = processor.process_queries([question]).to(model.device)

    with torch.inference_mode():
        question_embeddings = model(**inputs)[0].float().cpu()

    conn = database.create_connection()

    try:
        register_vector(conn)

        with conn.cursor() as curr:
            curr.execute("""
                SELECT
                    p.id, p.docid, p.page_num, p.text,
                    p.ispoisoned, d.filepath, e.embedding
                FROM pages p
                JOIN documents d ON d.id = p.docid
                JOIN page_embeddings e ON e.page_id = p.id
                ORDER BY p.id, e.vector_position
            """)

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
            results.append(page)

    results.sort(key=lambda page: page["score"], reverse=True)

    if not results:
        return []

    # Include every page tied with the kth page.
    cutoff = results[min(top_k, len(results)) - 1]["score"]

    selected = [
        page for page in results
        if page["score"] >= cutoff
    ]

    for rank, page in enumerate(selected, start=1):
        page["retrieval_rank"] = rank

    return selected