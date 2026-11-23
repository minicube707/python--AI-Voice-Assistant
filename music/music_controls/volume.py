import re

VOLUME_PATTERN = re.compile(r"(\d+)\s*(?:%|pour cent)?")


def extract_volume(sentence: str) -> int | None:
    """Return the first number found in the sentence, or None."""

    match = VOLUME_PATTERN.search(sentence)

    return int(match.group(1)) if match else None
