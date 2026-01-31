from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json
import re
from typing import Dict, Any
from difflib import SequenceMatcher

MODEL_NAME = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def parse_json(text: str) -> Dict[str, Any]:
    """
    Robustly parse JSON from LLM output
    """
    
    # Try 1: Direct JSON parsing
    try:
        # return json.loads(text)
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        elif isinstance(parsed, int):
            return {
                'accuracy': parsed,
                'completeness': parsed,
                'relevance': parsed,
                'coherence': parsed,
                'explanation': 'Single-score output from LLM'
            }

    except json.JSONDecodeError:
        pass
    
    # Try 2: Find JSON object in text
    try:
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
    except:
        pass
    
    # Try 3: Extract numbers from text
    try:
        numbers = re.findall(r'\b[1-5]\b', text)
        if len(numbers) >= 4:
            return {
                'accuracy': int(numbers[0]),
                'completeness': int(numbers[1]),
                'relevance': int(numbers[2]),
                'coherence': int(numbers[3]),
                'explanation': text[:200]
            }
    except:
        pass
    
    # Default fallback
    return {
        'accuracy': 2,
        'completeness': 2,
        'relevance': 2,
        'coherence': 2,
        'explanation': text[:100]
    }


def string_similarity(text1: str, text2: str) -> float:
    """
    Compute string-level similarity (0-1)
    """
    if not text1 or not text2:
        return 0.0
    
    ratio = SequenceMatcher(
        None,
        text1.lower()[:200],  # Compare first 200 chars
        text2.lower()[:200]
    ).ratio()
    
    return ratio


# def is_hallucinated(prediction: str, ground_truth: str) -> bool:
#     """
#     Detect hallucination using multiple signals
#     """
#     pred_lower = prediction.lower()
    
#     # Signal 1: Says it doesn't know (GOOD - not hallucinated)
#     no_knowledge_phrases = [
#         "i don't know",
#         "not sure",
#         "unclear",
#         "cannot determine",
#         "not mentioned",
#         "not provided",
#         "no information",
#         "cannot answer"
#     ]
#     if any(phrase in pred_lower for phrase in no_knowledge_phrases):
#         return False
    
#     # Signal 2: Very different from ground truth (BAD - likely hallucinated)
#     similarity = string_similarity(ground_truth, prediction)
#     if similarity < 0.15:  # Almost no overlap
#         return True
    
#     # Signal 3: Contains contradictions
#     if any(phrase in pred_lower for phrase in ["however", "but", "contrary"]):
#         if "contrary to" in pred_lower or "contradicts" in pred_lower:
#             return True
    
#     # Signal 4: Impossibly detailed when context is vague
#     if len(pred_lower.split()) > len(ground_truth.split()) * 2:
#         if similarity < 0.3:
#             return True
    
#     return False

def combine_scores(score1: float, score2: float, weights: list) -> float:
    """
    Weighted average of two scores
    """
    w1, w2 = weights
    combined = (score1 * w1 + score2 * w2) / (w1 + w2)
    return max(1.0, min(5.0, combined))


def llm_evaluate(
        question: str,
        ground_truth: str,
        prediction: str
    ) -> Dict[str, Any]:
        """
        LLM evaluation with STRICT rubric
        """
        
        # STRICT EVALUATION PROMPT
        prompt = f"""You are an EXPERT and STRICT evaluator. Your job is to rate answers HONESTLY.

RUBRIC - Be HARSH and HONEST:

Accuracy (1-5):
- 5 = PERFECTLY matches ground truth, no errors
- 4 = Mostly correct with minor omissions
- 3 = Partially correct (50%+ accurate)
- 2 = Mostly wrong with some correct parts
- 1 = COMPLETELY WRONG, hallucinated, or contradicts ground truth

Completeness (1-5):
- 5 = Covers ALL key aspects of the question
- 4 = Covers most important aspects
- 3 = Covers some aspects
- 2 = Covers few aspects
- 1 = Barely addresses the question

Relevance (1-5):
- 5 = Directly and fully answers the question
- 3 = Somewhat relevant but incomplete
- 1 = Not relevant or off-topic

Coherence (1-5):
- 5 = Very clear, well-organized, easy to understand
- 3 = Acceptable clarity
- 1 = Confusing, poorly written

---

GROUND TRUTH (what the answer should be):
{ground_truth[:200]}

QUESTION:
{question}

PREDICTION (the answer to evaluate):
{prediction[:200]}

---

IMPORTANT RULES:
1. If prediction contradicts ground truth, give 1-2 for accuracy
2. If prediction is hallucinated or makes up facts, give 1 for accuracy
3. Compare prediction to ground truth - if prediction is worse, give LOW scores
4. Be STRICT - only give 4-5 for truly excellent answers
5. If prediction ignores the context, give low scores

Rate HONESTLY and STRICTLY:

{{
  "accuracy": <1 to 5>,
  "completeness": <1 to 5>,
  "relevance": <1 to 5>,
  "coherence": <1 to 5>,
  "explanation": "<Why these scores? Be concise.>"
}}"""
        
        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=300,
                    num_beams=4,
                    temperature=0.5,  # More deterministic
                    do_sample=False
                )
            
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Try to parse JSON
            result = parse_json(output_text)

            if not isinstance(result, dict):
                raise ValueError(f"LLM judge returned non-dict output: {result}")
            
            # Validate scores
            for key in ['accuracy', 'completeness', 'relevance', 'coherence']:
                score = int(result.get(key, 3))
                result[key] = max(1, min(5, score))  # Clamp to 1-5
            
            return result
            
        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
            return {
                'accuracy': 2,
                'completeness': 2,
                'relevance': 2,
                'coherence': 2,
                'explanation': 'Failed to evaluate'
            }

def judge_answer(question: str, ground_truth: str, prediction: str) -> Dict[str, Any]:
    """
    Judge the generated answer using Flan-T5
    """
    
    # print(f"Judging answer for question: {question}")
    # print(f"Ground Truth: {ground_truth}")
    # print(f"Prediction: {prediction}")

    # hallucinated = is_hallucinated(prediction, ground_truth)

    string_sim = string_similarity(ground_truth, prediction)

    llm_scores = llm_evaluate(question, ground_truth, prediction)

    combined = {
            'accuracy': combine_scores(
                llm_scores['accuracy'],           # 70% weight
                string_sim * 5.0,                 # 30% weight (convert 0-1 to 0-5)
                weights=[0.7, 0.3]
            ),
            'completeness': llm_scores['completeness'],
            'relevance': llm_scores['relevance'],
            'coherence': llm_scores['coherence'],
            'explanation': llm_scores['explanation']
        }
    
    # if hallucinated:
    #     combined['accuracy'] = min(combined['accuracy'], 2)  # Max 2/5 if hallucinated
    
    return combined