from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def _empty_judge_result(reason="Invalid LLM judge output"):
    return {
        "accuracy": 0,
        "completeness": 0,
        "relevance": 0,
        "coherence": 0,
        "explanation": reason
    }

def judge_answer(question, ground_truth, prediction):
    """
    Returns scores (1–5) and explanation for each dimension
    """
    prompt = f"""
You are an expert evaluator for a Question Answering system.

Evaluate the generated answer using the following criteria:
1. Factual Accuracy (1–5)
2. Completeness (1–5)
3. Relevance (1–5)
4. Coherence (1–5)

Question:
{question}

Ground Truth Answer:
{ground_truth}

Generated Answer:
{prediction}

Respond strictly in JSON format:
{{
  "accuracy": <int>,
  "completeness": <int>,
  "relevance": <int>,
  "coherence": <int>,
  "explanation": "<short explanation>"
}}
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=300)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    try:
        parsed = json.loads(text)

        if isinstance(parsed, int):
            return _empty_judge_result(
                reason=f"LLM returned scalar instead of JSON: {parsed}"
            )

        if not isinstance(parsed, dict):
            return _empty_judge_result(
                reason=f"LLM returned invalid JSON type: {type(parsed)}"
            )

        # Ensure all required keys exist
        for key in ["accuracy", "completeness", "relevance", "coherence", "explanation"]:
            if key not in parsed:
                parsed[key] = 0 if key != "explanation" else "Missing field"

        return parsed
    except Exception as e:
        return _empty_judge_result(reason=f"Judge parse failure: {str(e)}")