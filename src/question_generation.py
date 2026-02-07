import json
import random
import re

DATA_PATH = "data/corpus_chunks.json"
OUTPUT_PATH = "data/questions_100.json"
random.seed(42)

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

def get_concise_answer(text, sentence_count=2):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return " ".join(sentences[:sentence_count])

questions = []
qid = 0

# =====================================================
# FACTUAL (30)
# =====================================================
print("Generating FACTUAL questions...")
for url in random.sample(urls, 30):
    c = random.choice(docs[url])

    short_answer = get_concise_answer(c["text"], 1)
    questions.append({
        "id": qid,
        "question": f"Based on the article '{c['title']}', what is the primary definition or role described?",
        "answer": short_answer,
        "question_type": "factual",
        "source_urls": [url],
        "source_chunk_ids": [c["chunk_id"]]
    })
    qid += 1

# =====================================================
# COMPARATIVE (15)
# =====================================================
print("Generating COMPARATIVE questions...")
for _ in range(15):
    u1, u2 = random.sample(urls, 2)
    c1, c2 = random.choice(docs[u1]), random.choice(docs[u2])

    questions.append({
        "id": qid,
        "question": f"Compare the focus of {c1['title']} with that of {c2['title']}. How do they differ?",
        "answer": f"{c1['title']} discusses: {get_concise_answer(c1['text'], 1)} Whereas {c2['title']} focuses on: {get_concise_answer(c2['text'], 1)}",
        "question_type": "comparative",
        "source_urls": [u1, u2],
        "source_chunk_ids": [c1["chunk_id"], c2["chunk_id"]]
    })
    qid += 1

# =====================================================
# INFERENTIAL (10)
# =====================================================
print("Generating INFERENTIAL questions...")
for _ in range(10):
    u = random.choice(urls)
    c = random.choice(docs[u])

    questions.append({
        "id": qid,
        "question": f"Given the details about {c['title']}, what can be inferred about its impact on its respective field?",
        "answer": f"Based on the context: '{c['text'][:200]}...', it can be inferred that {c['title']} plays a critical role in shaping outcomes by providing the necessary framework or data mentioned.",
        "question_type": "inferential",
        "source_urls": [u],
        "source_chunk_ids": [c["chunk_id"]]
    })
    qid += 1

# =====================================================
# MULTI-HOP (15)
# =====================================================
print("Generating MULTI-HOP questions...")
for _ in range(15):
    u1, u2 = random.sample(urls, 2)
    c1, c2 = random.choice(docs[u1]), random.choice(docs[u2])

    questions.append({
        "id": qid,
        "question": f"Using information from both {c1['title']} and {c2['title']}, identify a common thread or potential intersection between these two topics.",
        "answer": f"The connection lies in their shared relevance to the broader context provided in the corpus. {c1['title']} provides the foundation for {c1['text'][:50]}, while {c2['title']} expands on {c2['text'][:50]}.",
        "question_type": "multi-hop",
        "source_urls": [u1, u2],
        "source_chunk_ids": [c1["chunk_id"], c2["chunk_id"]]
    })
    qid += 1

# =====================================================
# AMBIGUOUS QUESTIONS (10)
# =====================================================
print("Generating AMBIGUOUS questions...")
for _ in range(10):
    u = random.choice(urls)
    c = random.choice(docs[u])

    questions.append({
        "id": qid,
        "question": f"Tell me more about the 'various factors' involved in {c['title']}.",
        "answer": "This is an ambiguous query. While the text mentions the subject, it does not provide a specific list of 'various factors' without further clarification on the aspect of interest.",
        "question_type": "ambiguous",
        "source_urls": [u],
        "source_chunk_ids": [c["chunk_id"]]
    })
    qid += 1

# =====================================================
# NEGATED QUESTIONS (10)
# =====================================================
print("Generating NEGATED questions...")
for _ in range(10):
    u = random.choice(urls)
    c = random.choice(docs[u])

    questions.append({
        "id": qid,
        "question": f"Which modern criticisms of {c['title']} are NOT addressed in the text?",
        "answer": f"The text focuses on the definition and history of {c['title']} but fails to mention contemporary criticisms or modern-day controversies.",
        "question_type": "negated",
        "source_urls": [u],
        "source_chunk_ids": [c["chunk_id"]]
    })
    qid += 1

# =====================================================
# PARAPHRASED ADVERSARIAL (10)
# =====================================================
print("Generating PARAPHRASED ADVERSARIAL questions...")
paraphrase_templates = [
    "Can you explain the concept of {title} in your own words?",
    "How would you describe {title} to someone who has never heard of it?",
    "What does the term '{title}' refer to in this context?",
    "Provide a brief overview of what {title} entails.",
    "Explain the meaning of {title} based on the provided text.",
    "Give me a summary of the information regarding {title}.",
    "What is meant by the phrase '{title}'?",
    "Describe the nature of {title} using the available information.",
]

for i in range(10):
    u = random.choice(urls)
    c = random.choice(docs[u])

    # Select a random template and format it with the chunk title
    question_text = random.choice(paraphrase_templates).format(title=c['title'])
    
    # The answer is the direct text from the chunk, demonstrating that the
    # model can link the paraphrased question back to the source.
    answer = c["text"]

    questions.append({
        "id": qid,
        "question": question_text,
        "answer": answer,
        "question_type": "paraphrased_adversarial",
        "source_urls": [u],
        "source_chunk_ids": [c["chunk_id"]]
    })
    qid += 1

# =====================================================
# UNANSWERABLE / HALLUCINATION (10)
# =====================================================
print("Generating UNANSWERABLE / HALLUCINATION questions...")
for i in range(10):
    u = random.choice(urls)
    c = random.choice(docs[u])

    # These are designed to look like factual questions but they ask for 
    # data that is logically impossible to find in a Wikipedia chunk.
    impossible_questions = [
        f"What is the secret, unpublished password for the {c['title']} system?",
        f"Which specific individual wrote the 4th sentence of the {c['title']} Wikipedia entry?",
        f"On what exact date will {c['title']} be officially discontinued in the year 2099?",
        f"What is the GPS coordinate of the exact center of {c['title']}?"
    ]

    question_text = random.choice(impossible_questions)
    answer = "This information is not available in the provided context. The document does not contain internal system data, future predictions, or meta-data about its own authorship."

    questions.append({
        "id": qid,
        "question": question_text,
        "answer": answer,
        "question_type": "unanswerable",
        "source_urls": [u],
        "source_chunk_ids": [c["chunk_id"]]
    })
    qid += 1

# =====================================================
# SAVE
# =====================================================
with open(OUTPUT_PATH, "w") as f:
    json.dump(questions, f, indent=2)

print(f"Generated {len(questions)} challenging evaluation questions")
