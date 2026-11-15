"""Generate answers from page images; leave correctness to the human reviewer."""
from functools import lru_cache

from prompts import GENERATOR_PROMPT, NO_CONTEXT_PROMPT

import pymupdf
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration



_settings = {}

def configure(settings):
    global _settings
    if settings != _settings:
        setup.cache_clear()
    _settings = dict(settings)


@lru_cache(maxsize=1)
def setup():
    spec = _settings.get("models", {}).get("generator", {})
    model_name = spec.get("id", "Qwen/Qwen2.5-VL-7B-Instruct")

    if torch.cuda.is_available():
        device = "cuda"
        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        revision=spec.get("revision"),
        torch_dtype=dtype,
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(model_name, revision=spec.get("revision"))

    return model, processor


def generate_answer(question, pages, max_new_tokens=256, allow_no_context=False):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if not pages and not allow_no_context:
        return "No relevant pages were found."

    print(
        "Generator sources:",
        [(page["filepath"], page["page_num"]) for page in pages],
        flush=True,
    )

    model, processor = setup()
    images = []
    content = []

    for source_number, page in enumerate(pages, start=1):
        with pymupdf.open(page["filepath"]) as pdf:
            image = pdf[page["page_num"] - 1].get_pixmap(
                dpi=_settings.get("dpi", 150),
                alpha=False,
            ).pil_image().convert("RGB")

        images.append(image)

        content.extend([
            {
                "type": "text",
                "text": (
                    f"Source [{source_number}]: "
                    f"document {page['document_id']}, "
                    f"page {page['page_num']}."
                ),
            },
            {"type": "image", "image": image},
        ])

    content.append({
        "type": "text",
        "text": f"Question: {question}",
    })

    messages = [
        {
            "role": "system",
            "content": GENERATOR_PROMPT if pages else NO_CONTEXT_PROMPT,
        },
        {"role": "user", "content": content},
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    input_kwargs = {"text": [text], "return_tensors": "pt"}
    if images:
        input_kwargs["images"] = images
    inputs = processor(**input_kwargs).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    answer_tokens = output[:, inputs["input_ids"].shape[1]:]

    return processor.batch_decode(
        answer_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
