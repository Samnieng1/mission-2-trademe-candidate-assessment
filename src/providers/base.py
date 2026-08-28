"""Base provider interface and helpers."""
from __future__ import annotations

from abc import ABC, abstractmethod
import json
import re
from typing import Any, Optional

from ..schemas import ProviderResult


class ModelProvider(ABC):
    """Abstract provider interface for model adapters."""

    @abstractmethod
    def analyse(self, job_description: str, candidate_profile: str) -> ProviderResult:
        """Analyse the supplied texts and return a ProviderResult."""

    @abstractmethod
    def health(self) -> bool:
        """Return True if provider is configured and reachable (best-effort)."""


def extract_json_payload(raw_text: Optional[str]) -> Optional[Any]:
    """Best-effort extraction of a JSON payload from model output text."""
    if not raw_text:
        return None

    text = raw_text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        candidate = fenced_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    first_object = text.find("{")
    last_object = text.rfind("}")
    if first_object != -1 and last_object != -1 and last_object > first_object:
        candidate = text[first_object:last_object + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    first_array = text.find("[")
    last_array = text.rfind("]")
    if first_array != -1 and last_array != -1 and last_array > first_array:
        candidate = text[first_array:last_array + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None



class BaseProvider(ModelProvider):
    """Alias for the provider base class to match suggested naming.

    Implementations should provide `analyse(...)` and `health()` methods and
    return `ProviderResult` objects. This alias preserves the existing
    `ModelProvider` contract while offering the `BaseProvider` name suggested
    in docs.
    """
    pass

