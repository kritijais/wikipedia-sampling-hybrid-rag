# !pip install wikipedia-api
import requests
import random
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Wikipedia API endpoint
WIKI_API = "https://en.wikipedia.org/w/api.php"

# HTTP headers for requests
HEADERS = {
    "User-Agent": "HybridRAG-Group7/1.0"
}

# Category-based random sampling
RANDOM_CATEGORIES = {
    "Biology": "Category:Biology",
    "Physics": "Category:Physics",
    "Chemistry": "Category:Chemistry",
    "Economics": "Category:Economics",
    "Psychology": "Category:Psychology",
    "History": "Category:History",
    "Geography": "Category:Geography",
    "Music": "Category:Music",
    "Literature": "Category:Literature",
    "Technology": "Category:Technology",
    "Politics": "Category:Politics",
    "Space": "Category:Space"
}

# Configuration
MAX_SUBCATS_PER_CATEGORY = 3
MAX_PAGES_PER_CATEGORY = 120

# Fetch pages in a category
def get_category_pages(category, limit=300):
    pages = []
    cmcontinue = None

    while len(pages) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": 100,
            "format": "json"
        }

        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            break

        data = r.json()

        for p in data.get("query", {}).get("categorymembers", []):
            if p["ns"] == 0:
                pages.append(
                    f"https://en.wikipedia.org/wiki/{p['title'].replace(' ', '_')}"
                )

        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break

        time.sleep(0.5)

    return pages[:limit]

# Fetch subcategories of a category
def get_subcategories(category):
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmtype": "subcat",
        "cmlimit": 50,
        "format": "json"
    }
    r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
    if r.status_code != 200: # Add check for non-200 status code
        print(f"Warning: Failed to fetch subcategories for {category}. Status code: {r.status_code}")
        return []

    try:
        data = r.json()
    except requests.exceptions.JSONDecodeError as e: # Catch JSON decoding error
        print(f"Warning: Could not decode JSON for subcategories of {category}. Error: {e}. Response text: {r.text[:200]}")
        return []

    return [c["title"] for c in data.get("query", {}).get("categorymembers", [])]

# Sample random URLs using category-based sampling
def sample_random_urls(target=300):
    random_urls = set()
    categories = list(RANDOM_CATEGORIES.values())
    random.shuffle(categories)

    for cat in categories:
        pages = get_category_pages(cat, limit=MAX_PAGES_PER_CATEGORY)
        random_urls.update(pages)

        subcats = get_subcategories(cat)[:MAX_SUBCATS_PER_CATEGORY]
        for sc in subcats:
            sub_pages = get_category_pages(sc, limit=80)
            random_urls.update(sub_pages)

        if len(random_urls) >= target * 2:
            break

    random_urls = list(random_urls)
    random.shuffle(random_urls)
    random_urls = random_urls[:target]

    # Save sampled random URLs for reference
    with open("data/random_urls.json", "w") as f:
        json.dump({
            "type": "random",
            "sampling_method": "category_based_random_sampling",
            "total_urls": len(random_urls),
            "urls": random_urls
        }, f, indent=2)

    return random_urls

# Fetch page text from Wikipedia API
def fetch_page_text(url, min_words=200, retries=3):
    title = url.split("/wiki/")[-1]

    # Skip non-article pages early
    if any(b in title for b in ["List_of", "Outline_of", "(disambiguation)"]):
        return None

    # API request parameters
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": True,
        "redirects": True,
        "titles": title
    }

    # Retry logic for robustness
    for _ in range(retries):
        try:
            r = requests.get(
                WIKI_API,
                params=params,
                headers=HEADERS,
                timeout=20
            )

            if r.status_code != 200:
                time.sleep(1)
                continue

            data = r.json()
            pages = data.get("query", {}).get("pages", {})

            for _, page in pages.items():
                if "extract" not in page:
                    return None

                text = page["extract"].strip()
                if len(text.split()) < min_words:
                    return None

                return {
                    "url": url,
                    "title": page.get("title", title),
                    "text": text
                }

        except Exception:
            time.sleep(1)

    return None

# Build corpus until target number of documents is reached
def build_corpus(urls, target_docs=500, max_workers=8):
    corpus = []
    seen_urls = set()

    # Use ThreadPoolExecutor for concurrent fetching
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_page_text, url): url
            for url in urls
        }

        for future in as_completed(futures):
            if len(corpus) >= target_docs:
                break

            url = futures[future]
            if url in seen_urls:
                continue

            seen_urls.add(url)

            page = future.result()
            if page:
                corpus.append(page)

    return corpus

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # Load fixed URLs
    with open("data/fixed_urls.json") as f:
        fixed_urls = json.load(f)

    target_docs = 500
    corpus = []
    seen_urls = set()

    # Always include fixed URLs
    base_urls = fixed_urls["urls"].copy()

    while len(corpus) < target_docs:
        needed = target_docs - len(corpus)

        # Oversample aggressively
        random_urls = sample_random_urls(target=max(800, needed * 3))
        all_urls = base_urls + random_urls
        random.shuffle(all_urls)

        print(f" Attempting with {len(all_urls)} URLs...")

        # Build corpus
        new_docs = build_corpus(
            all_urls,
            target_docs=target_docs - len(corpus),
            max_workers=8
        )

        for doc in new_docs:
            if doc["url"] not in seen_urls:
                corpus.append(doc)
                seen_urls.add(doc["url"])

        print(f" Corpus size now: {len(corpus)}")

        if len(new_docs) == 0:
            print(" No new documents found, stopping to avoid infinite loop.")
            break

    # Save corpus to file
    with open("data/raw_corpus.json", "w") as f:
        json.dump({
            "total_documents": len(corpus),
            "fixed_urls": len(base_urls),
            "random_urls_sampled": len(random_urls),
            "documents": corpus
        }, f, indent=2)

    print(f" Corpus created with {len(corpus)} documents")