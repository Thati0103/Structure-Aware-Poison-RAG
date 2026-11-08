from functools import lru_cache

import pymupdf
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


@lru_cache(maxsize=1)
def setup():
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "lightonai/MonoQwen2-VL-v0.1",
        torch_dtype=dtype,
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct"
    )

    return model, processor


def score_page(question, image, model, processor):
    prompt = (
        "Assert the relevance of the previous image document to the "
        "following query, answer True or False. The query is: "
        + question
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    true_id = processor.tokenizer.convert_tokens_to_ids("True")
    false_id = processor.tokenizer.convert_tokens_to_ids("False")

    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1, :]
        score = torch.softmax(
            logits[[true_id, false_id]].float(),
            dim=-1,
        )[0].item()

    return score


def rerank_pages(question, pages, top_k=3):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    if not pages:
        return []

    model, processor = setup()
    results = []

    for retrieval_rank, page in enumerate(pages, start=1):
        with pymupdf.open(page["filepath"]) as pdf:
            # Database page numbers start at 1; PyMuPDF starts at 0.
            pdf_page = pdf[page["page_num"] - 1]
            image = pdf_page.get_pixmap(
                dpi=150,
                alpha=False,
            ).pil_image().convert("RGB")

        score = score_page(question, image, model, processor)

        results.append({
            **page,
            "retrieval_rank": retrieval_rank,
            "reranker_score": score,
        })

    results.sort(
        key=lambda page: page["reranker_score"],
        reverse=True,
    )

    for rank, page in enumerate(results, start=1):
        page["reranker_rank"] = rank

    return results[:top_k]