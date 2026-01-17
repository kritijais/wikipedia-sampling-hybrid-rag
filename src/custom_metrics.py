import spacy
from nltk.translate.bleu_score import sentence_bleu
import random

nlp = spacy.load("en_core_web_sm")

# Entity Coverage Score (ECS)
def extract_entities(text):
    doc = nlp(text)
    return set(ent.text.lower() for ent in doc.ents)

def entity_coverage_score(ground_truth, prediction):
    gt_entities = extract_entities(ground_truth)
    if not gt_entities:
        return 1.0  # nothing to miss
    pred_entities = extract_entities(prediction)
    return round(len(gt_entities & pred_entities) / len(gt_entities), 4)


# Answer Diversity Score (ADS)
def answer_diversity(answers, sample_size=50):
    if len(answers) < 2:
        return 1.0

    scores = []
    sampled = random.sample(answers, min(sample_size, len(answers)))

    for i, a in enumerate(sampled):
        refs = [s.split() for j, s in enumerate(sampled) if j != i]
        score = sentence_bleu(refs, a.split())
        scores.append(score)

    return round(1 - sum(scores) / len(scores), 4)

# Hallucination Rate (HR)
def is_hallucinated(rank, entity_coverage, threshold=0.3):
    if rank == 0:
        return True
    if entity_coverage < threshold:
        return True
    return False

def hallucination_rate(flags):
    return round(sum(flags) / len(flags), 4)

# Temporal Consistency Score (TCS)
def temporal_consistency(answers):
    if len(answers) < 2:
        return 1.0

    scores = []
    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            score = sentence_bleu(
                [answers[j].split()],
                answers[i].split()
            )
            scores.append(score)

    return round(sum(scores) / len(scores), 4)
