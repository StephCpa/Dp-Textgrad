from __future__ import annotations

"""
Utility helpers for handling textual feedback in a privacy-aware way.

These helpers enforce length limits and lightweight filtering so that
downstream components can reason about the sensitivity of textual
signals before they are logged or fed back into generation prompts.
"""

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class FeedbackSanitiserConfig:
    """Configuration for trimming and filtering textual feedback."""

    max_length: int = 512
    allowed_characters: Optional[str] = None
    replacement_character: str = "?"

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ValueError("max_length must be positive.")
        if self.allowed_characters is not None and len(self.allowed_characters) == 0:
            raise ValueError("allowed_characters cannot be empty when provided.")


class FeedbackSanitiser:
    """Enforce basic constraints on textual feedback before reuse."""

    def __init__(self, config: Optional[FeedbackSanitiserConfig] = None) -> None:
        self.config = config or FeedbackSanitiserConfig()

    def sanitise(self, text: str) -> str:
        """Trim and filter a text string to satisfy the configured policy."""
        truncated = text[: self.config.max_length]
        if self.config.allowed_characters is None:
            return truncated
        allowed = set(self.config.allowed_characters)
        replacement = self.config.replacement_character
        return "".join(ch if ch in allowed else replacement for ch in truncated)

    def batch(self, texts: Iterable[str]) -> list[str]:
        """Apply sanitisation to an iterable of strings."""
        return [self.sanitise(text) for text in texts]

