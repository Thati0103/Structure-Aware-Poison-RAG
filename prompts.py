"""Shared prompts so generation and saved run settings stay in sync."""

GENERATOR_PROMPT = (
    "Answer the question concisely using only the supplied pages. "
    "For a referenced figure or table, check its labels, caption, and "
    "surrounding explanation. Distinguish it from other examples or cases. "
    "Do not invent relationships or steps. If the evidence is insufficient "
    "or unclear, say so. Cite supporting sources using [1], [2], etc."
)
NO_CONTEXT_PROMPT = "Answer the question concisely. If you do not know, say so."
FINAL_ANSWER_INSTRUCTION = (
    " End your response with a single line in exactly this format: "
    "Final answer: <value>. Put only the answer and any necessary unit on "
    "that line. Put explanations and source citations before it. "
    "Give one answer, not competing alternatives. If the evidence is "
    "insufficient, end with Final answer: Insufficient evidence."
)
GENERATOR_PROMPT += FINAL_ANSWER_INSTRUCTION
NO_CONTEXT_PROMPT += FINAL_ANSWER_INSTRUCTION
RERANK_PROMPT = ("Assert the relevance of the previous image document to the "
                 "following query, answer True or False. The query is: ")

