"""Conservative text normalization for extracted document content."""

import re


def clean_text(text: str) -> str:
    """Normalize line endings and whitespace without removing paragraph breaks.

    Horizontal whitespace (spaces and tabs) is collapsed within each line;
    leading and trailing whitespace is removed from lines and the full text.
    Any run of two or more blank lines is reduced to one blank line.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n(?:[ \t]*\n){2,}", "\n\n", cleaned)
    return cleaned.strip()
