from typing import List, Optional
import re
import math
import random

# -------------------- STOPWORDS --------------------
_DEFAULT_STOPWORDS = {
    "a","an","and","are","as","at","be","by","for","from",
    "has","he","in","is","it","its","of","on","that","the",
    "to","was","were","will","with",
}

# -------------------- HELPERS --------------------
def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())

# -------------------- BASIC EXTRACTIVE --------------------
def summarize_extractive(text: str, num_sentences: Optional[int] = None, ratio: float = 0.2) -> str:
    if not text.strip():
        return ""

    sentences = _split_sentences(text)
    if not sentences:
        return ""

    if num_sentences is None:
        num_sentences = max(1, math.ceil(len(sentences) * ratio))

    freq = {}
    for sent in sentences:
        for word in _tokenize_words(sent):
            if word in _DEFAULT_STOPWORDS:
                continue
            freq[word] = freq.get(word, 0) + 1

    if not freq:
        return " ".join(sentences[:num_sentences])

    max_freq = max(freq.values())
    for w in freq:
        freq[w] /= max_freq

    scores = []
    for i, sent in enumerate(sentences):
        words = _tokenize_words(sent)
        if not words:
            scores.append((i, 0))
            continue
        score = sum(freq.get(w, 0) for w in words) / len(words)
        scores.append((i, score))

    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    top_indices = sorted([i for i, _ in scores_sorted[:num_sentences]])

    return " ".join([sentences[i] for i in top_indices])

# -------------------- VARIED (RANDOM) SUMMARIZER --------------------
def summarize_extractive_varied(text: str, num_sentences=None, ratio=0.2, diversity=0.4):
    if not text.strip():
        return ""

    sentences = _split_sentences(text)
    if not sentences:
        return ""

    if num_sentences is None:
        num_sentences = max(1, math.ceil(len(sentences) * ratio))

    freq = {}
    for sent in sentences:
        for word in _tokenize_words(sent):
            if word in _DEFAULT_STOPWORDS:
                continue
            freq[word] = freq.get(word, 0) + 1

    if not freq:
        return " ".join(sentences[:num_sentences])

    max_freq = max(freq.values())
    for w in freq:
        freq[w] /= max_freq

    scores = []
    for i, sent in enumerate(sentences):
        words = _tokenize_words(sent)
        if not words:
            scores.append((i, 0))
            continue
        score = sum(freq.get(w, 0) for w in words) / len(words)
        scores.append((i, score))

    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)

    # 🔥 randomness
    top_k = max(num_sentences + 2, int(len(sentences) * diversity))
    candidates = scores_sorted[:top_k]

    chosen = random.sample(candidates, num_sentences)
    top_indices = sorted([i for i, _ in chosen])

    return " ".join([sentences[i] for i in top_indices])

# -------------------- ABSTRACTIVE (AI) --------------------
def summarize_abstractive(text: str, model_name="facebook/bart-large-cnn"):
    try:
        from transformers import pipeline
    except:
        return "Install transformers: pip install transformers torch"

    summarizer = pipeline("summarization", model=model_name)

    result = summarizer(
        text,
        max_length=60,
        min_length=20,
        do_sample=True,     # 🔥 gives different summaries
        temperature=0.9
    )

    return result[0]['summary_text']

# -------------------- MAIN TEST --------------------
if __name__ == "__main__":
    text = """Artificial Intelligence (AI) is transforming industries by enabling machines to perform 
    tasks that typically require human intelligence. It is widely used in healthcare for disease prediction,
      in finance for fraud detection, and in transportation for autonomous vehicles. Despite its advantages,
        AI also raises concerns about job displacement and ethical issues. Experts believe that proper regulation
          and responsible development are essential to maximize its benefits while minimizing risks."""

    print("\n---  SUMMARY ---")
    print(summarize_extractive(text, num_sentences=2))

    