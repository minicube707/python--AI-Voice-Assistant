"""
neural/preprocessing.py
-----------------------
Text preprocessing shared by every neural model of the project:
tokenization, stemming and bag-of-words encoding.
"""

from __future__ import annotations

import nltk
import numpy as np
from nltk.stem.snowball import SnowballStemmer

from nlp.utils import normalize

IGNORED_CHARACTERS = {"?", "!", ".", ","}

# Every number is replaced by this token so that "volume à 30" and
# "volume à 50" share the same representation.
NUMBER_TOKEN = "<num>"

_NLTK_RESOURCES = {
    "tokenizers/punkt": "punkt",
    "tokenizers/punkt_tab": "punkt_tab",
}


def _download_resources() -> None:
    for resource, package in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)


class TextPreprocessor:
    """Turn raw French text into tokens and bag-of-words vectors."""

    def __init__(self, language: str = "french"):
        self.language = language
        self.stemmer = SnowballStemmer(language)
        _download_resources()

    def tokenize(self, text: str) -> list[str]:
        tokens = []

        for token in nltk.word_tokenize(text, language=self.language):
            if token in IGNORED_CHARACTERS:
                continue

            token = normalize(token)

            if not token:
                continue

            if token.isdigit():
                tokens.append(NUMBER_TOKEN)
            else:
                tokens.append(self.stemmer.stem(token))

        return tokens

    def bag_of_words(self, text: str, vocabulary: list[str]) -> np.ndarray:
        tokens = set(self.tokenize(text))

        return np.asarray(
            [float(word in tokens) for word in vocabulary],
            dtype=np.float32,
        )
