import unidecode

CORRECTIONS = {
    "mais": "mets",
    "memoire": "mets moi",
    "may": "mets",
    "memoir": "mets moi",
    "mes": "mets",
    "reprends": "reprend",
}

PUNCTUATION_TO_REMOVE = ".,'()-<>!?"
TRANSLATION_TABLE = str.maketrans(PUNCTUATION_TO_REMOVE, " " * len(PUNCTUATION_TO_REMOVE))


def normalize(text: str) -> str:
    """Normalizes text for reliable keyword matching.

    The normalization converts the text to lowercase, removes leading and
    trailing whitespace, replaces accented characters with their ASCII
    equivalents, and removes selected punctuation marks.

    Args:
        text (str): The text to normalize.

    Returns:
        str: The normalized text.
    """

    string = unidecode.unidecode(text.lower().strip()).translate(TRANSLATION_TABLE).strip()
    return " ".join(string.split()).strip()


def apply_prefix_corrections(text: str) -> str:
    """Applies transcription corrections to the beginning of a text.

    Checks whether the text starts with a known incorrect transcription and
    replaces it with the corresponding correction. Only the first matching
    correction is applied.

    Args:
        text (str): The text to correct.

    Returns:
        str: The corrected text.
    """

    text = text.strip()
    for wrong, correct in CORRECTIONS.items():
        
        if text.startswith(wrong):
            text = correct + text[len(wrong):]
            break

    return text