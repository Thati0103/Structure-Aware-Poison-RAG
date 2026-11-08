from functools import lru_cache

import pymupdf
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


@lru_cache(maxsize=1)
def setup():
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=dtype,
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(model_name)

    return model, processor


def generate_answer(question, pages, max_new_tokens=256):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if not pages:
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
                dpi=150,
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
          "content": (
              "Answer the question concisely using only the supplied pages. "
              "For a referenced figure or table, check its labels, caption, and "
              "surrounding explanation. Distinguish it from other examples or cases. "
              "Do not invent relationships or steps. If the evidence is insufficient "
              "or unclear, say so. Cite supporting sources using [1], [2], etc."
          ),
      },
      {"role": "user", "content": content},
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=images,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode only the answer, excluding the input prompt.
    answer_tokens = output[:, inputs["input_ids"].shape[1]:]

    return processor.batch_decode(
        answer_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()