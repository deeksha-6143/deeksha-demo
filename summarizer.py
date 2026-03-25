"""Simple text summarization utilities.

Provides a lightweight extractive summarizer with no external dependencies
and an optional abstractive summarizer wrapper that uses Hugging Face
transformers if installed.

Functions:
 - summarize_extractive(text, num_sentences=None, ratio=0.2)
 - summarize_abstractive(text, model_name='facebook/bart-large-cnn')

The extractive summarizer uses a frequency-based approach (like a
simplified TextRank) and is suitable for small projects or demos.
"""
from typing import List, Optional
import re
import math


_DEFAULT_STOPWORDS = {
	# small stopword set to avoid dependency on NLTK for simple tasks
	"a",
	"an",
	"and",
	"are",
	"as",
	"at",
	"be",
	"by",
	"for",
	"from",
	"has",
	"he",
	"in",
	"is",
	"it",
	"its",
	"of",
	"on",
	"that",
	"the",
	"to",
	"was",
	"were",
	"will",
	"with",
}


def _split_sentences(text: str) -> List[str]:
	# Simple sentence splitter using punctuation. Keeps sentence endings.
	# This is intentionally lightweight so there are no external deps.
	# It will not be perfect for all languages or edge cases.
	sentences = re.split(r'(?<=[.!?])\s+', text.strip())
	# Filter out empty sentences
	return [s.strip() for s in sentences if s.strip()]


def _tokenize_words(text: str) -> List[str]:
	# Lowercase, keep alphanumeric words
	return re.findall(r"\w+", text.lower())


def summarize_extractive(text: str, num_sentences: Optional[int] = None, ratio: float = 0.2) -> str:
	"""Return an extractive summary for `text`.

	Parameters:
	- text: input document
	- num_sentences: number of sentences to return; if None, computed by `ratio`
	- ratio: fraction of sentences to keep when `num_sentences` is None

	The algorithm:
	1. Split into sentences
	2. Build simple word frequency table (ignoring a small stopword set)
	3. Score sentences by sum of word frequencies divided by sentence length
	4. Pick top-scoring sentences and return them in original order
	"""
	if not text or not text.strip():
		return ""

	sentences = _split_sentences(text)
	if len(sentences) == 0:
		return ""

	if num_sentences is None:
		# at least 1 sentence
		num_sentences = max(1, math.ceil(len(sentences) * float(ratio)))

	# Build frequency table
	freq = {}
	for sent in sentences:
		for word in _tokenize_words(sent):
			if word in _DEFAULT_STOPWORDS:
				continue
			freq[word] = freq.get(word, 0) + 1

	if not freq:
		# fallback: return first N sentences
		selected = sentences[:num_sentences]
		return " ".join(selected)

	# normalize frequencies
	max_freq = max(freq.values())
	for w in list(freq.keys()):
		freq[w] = freq[w] / max_freq

	# score sentences
	scores = []
	for i, sent in enumerate(sentences):
		words = _tokenize_words(sent)
		if not words:
			scores.append((i, 0.0))
			continue
		score = sum(freq.get(w, 0.0) for w in words) / len(words)
		scores.append((i, score))

	# pick top N sentence indices
	scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
	top_indices = sorted([i for i, _ in scores_sorted[:num_sentences]])

	summary_sentences = [sentences[i] for i in top_indices]
	return " ".join(summary_sentences)


def summarize_abstractive(text: str, model_name: str = "facebook/bart-large-cnn", **kwargs) -> str:
	"""Abstractive summarization wrapper using Hugging Face transformers pipeline.

	This function will attempt to import transformers and run the summarization
	pipeline. If `transformers` is not installed, it raises an informative
	ImportError so users know how to install the optional dependency.

	Extra kwargs are forwarded to the pipeline call (e.g., max_length,
	min_length, do_sample).
	"""
	try:
		from transformers import pipeline
	except Exception as e:  # pragma: no cover - optional dependency
		raise ImportError(
			"transformers is required for abstractive summarization. "
			"Install with: pip install transformers torch"
		) from e

	summarizer = pipeline("summarization", model=model_name)
	# The pipeline accepts chunks of text; for long documents user should chunk.
	result = summarizer(text, **kwargs)
	if isinstance(result, list) and len(result) > 0 and "summary_text" in result[0]:
		return result[0]["summary_text"].strip()
	# fallback
	return str(result)


if __name__ == "__main__":
	# Quick manual test
	sample = (
		"Natural language processing (NLP) is a subfield of linguistics, computer science, "
		"and artificial intelligence concerned with the interactions between computers and human language, "
		"in particular how to program computers to process and analyze large amounts of natural language data. "
		"The result is a computer capable of 'understanding' the contents of documents, including the contextual nuances of the language within them."
	)
	print("Extractive summary:\n", summarize_extractive(sample, num_sentences=2))
