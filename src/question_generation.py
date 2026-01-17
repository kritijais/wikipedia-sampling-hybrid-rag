import json
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DATA_PATH = "data/corpus_chunks.json"
OUTPUT_PATH = "data/questions_100.json"
MODEL_NAME = "google/flan-t5-base"

random.seed(42)

# -----------------------------
# Load model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def generate_llm(prompt, max_len=150):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=max_len)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# Load corpus chunks
# -----------------------------
with open(DATA_PATH) as f:
    chunks = json.load(f)

# Group chunks by URL
docs = {}
for c in chunks:
    docs.setdefault(c["url"], []).append(c)

urls = list(docs.keys())

questions = []
qid = 0

# =====================================================
# FACTUAL (30)
# =====================================================
for url in random.sample(urls, 30):
    c = random.choice(docs[url])
    questions.append({
        "id": qid,
        "question": f"What is {c['title']}?",
        "answer": c["text"][:300],
        "question_type": "factual",
        "source_urls": [url],
        "source_chunk_ids": [c["chunk_id"]]
    })
    qid += 1

# =====================================================
# COMPARATIVE (15)
# =====================================================
for _ in range(15):
    u1, u2 = random.sample(urls, 2)
    c1, c2 = random.choice(docs[u1]), random.choice(docs[u2])

    answer = generate_llm(
        f"Context A:\n{c1['text']}\n\n"
        f"Context B:\n{c2['text']}\n\n"
        f"Compare {c1['title']} and {c2['title']}."
    )

    questions.append({
        "id": qid,
        "question": f"Compare {c1['title']} and {c2['title']}.",
        "answer": answer,
        "question_type": "comparative",
        "source_urls": [u1, u2],
        "source_chunk_ids": [c1["chunk_id"], c2["chunk_id"]]
    })
    qid += 1

# =====================================================
# INFERENTIAL (10)
# =====================================================
for _ in range(10):
    u = random.choice(urls)
    c = random.choice(docs[u])

    answer = generate_llm(
        f"Context:\n{c['text']}\n\nWhat can be inferred from this?"
    )

    questions.append({
        "id": qid,
        "question": "What can be inferred from this information?",
        "answer": answer,
        "question_type": "inferential",
        "source_urls": [u],
        "source_chunk_ids": [c["chunk_id"]]
    })
    qid += 1

# =====================================================
# MULTI-HOP (15)
# =====================================================
for _ in range(15):
    u1, u2 = random.sample(urls, 2)
    c1, c2 = random.choice(docs[u1]), random.choice(docs[u2])

    answer = generate_llm(
        f"Context A:\n{c1['text']}\n\n"
        f"Context B:\n{c2['text']}\n\n"
        f"How are these two topics related?"
    )

    questions.append({
        "id": qid,
        "question": f"How are {c1['title']} and {c2['title']} related?",
        "answer": answer,
        "question_type": "multi-hop",
        "source_urls": [u1, u2],
        "source_chunk_ids": [c1["chunk_id"], c2["chunk_id"]]
    })
    qid += 1

# =====================================================
# AMBIGUOUS QUESTIONS (10)
# =====================================================
for _ in range(10):
    u = random.choice(urls)
    c = random.choice(docs[u])

    questions.append({
        "id": qid,
        "question": f"Explain the significance of {c['title']}.",
        "answer": c["text"][:300],
        "question_type": "ambiguous",
        "source_urls": [u],
        "source_chunk_ids": [c["chunk_id"]]
    })
    qid += 1

# =====================================================
# NEGATED QUESTIONS (10)
# =====================================================
for _ in range(10):
    u = random.choice(urls)
    c = random.choice(docs[u])

    questions.append({
        "id": qid,
        "question": f"Which of the following is NOT true about {c['title']}?",
        "answer": c["text"][:300],
        "question_type": "negated",
        "source_urls": [u],
        "source_chunk_ids": [c["chunk_id"]]
    })
    qid += 1

# =====================================================
# PARAPHRASED ADVERSARIAL (10)
# =====================================================
paraphrases = [
    "Can you explain the concept of {}?",
    "How would you describe {}?",
    "What does {} refer to?",
    "Explain {} in simple terms.",
    "What is meant by {}?"
]

for _ in range(10):
    u = random.choice(urls)
    c = random.choice(docs[u])

    questions.append({
        "id": qid,
        "question": random.choice(paraphrases).format(c["title"]),
        "answer": c["text"][:300],
        "question_type": "paraphrased_adversarial",
        "source_urls": [u],
        "source_chunk_ids": [c["chunk_id"]]
    })
    qid += 1

# =====================================================
# UNANSWERABLE / HALLUCINATION (10)
# =====================================================
for _ in range(10):
    u = random.choice(urls)
    c = random.choice(docs[u])

    questions.append({
        "id": qid,
        "question": f"What is the Nobel Prize year of {c['title']}?",
        "answer": "Not answerable from the given context.",
        "question_type": "unanswerable",
        "source_urls": [],
        "source_chunk_ids": []
    })
    qid += 1

# =====================================================
# SAVE
# =====================================================
with open(OUTPUT_PATH, "w") as f:
    json.dump(questions, f, indent=2)

print(f"Generated {len(questions)} challenging evaluation questions")
