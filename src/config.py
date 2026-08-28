"""Configuration settings for the experiment.

Loads environment variables and provides a singleton `settings` object.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import os

from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseModel):
    """Application settings loaded from environment variables via `os.getenv()`.

    Using `BaseModel` instead of `BaseSettings` keeps the dependency surface
    compatible with pydantic v2 installations where `BaseSettings` lives in
    a separate package.
    """

    OPENAI_API_KEY: Optional[str] = Field(None)
    OPENAI_MODEL: str = Field("gpt-5")

    HF_TOKEN: Optional[str] = Field(None)
    HF_MODEL: str = Field("microsoft/Phi-4")

    REQUEST_TIMEOUT_SECONDS: int = Field(120)
    MAX_OUTPUT_TOKENS: int = Field(3000)
    EVIDENCE_FUZZY_THRESHOLD: float = Field(0.70)

    # Default results directory pinned to the project root `results/` folder
    RESULTS_DIRECTORY: Path = Field(Path(__file__).resolve().parents[1] / "results")

    PRICING: Optional[Dict[str, Dict[str, Any]]] = None
    # Mission 2 CV-only scoring weights mirrored from candidate_scoring.py.
    CANDIDATE_SCORING: Dict[str, float] = Field(
        default_factory=lambda: {"mandatory": 80.0, "preferred": 20.0}
    )
    # Legacy compatibility field retained for earlier task notes. Mission 2 does
    # not apply unsupported-claim penalties in the live CV-only scoring flow.
    CANDIDATE_PENALTY_PER_UNSUPPORTED: float = Field(0.0)

    @validator("RESULTS_DIRECTORY", pre=True)
    def _coerce_path(cls, v):
        if v is None:
            return Path(__file__).resolve().parents[1] / "results"
        return Path(v)


def get_settings() -> Settings:
    global _SETTINGS  # type: ignore

    try:
        return _SETTINGS  # type: ignore
    except NameError:
        _SETTINGS = Settings(
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
            OPENAI_MODEL=os.getenv("OPENAI_MODEL", "gpt-5"),
            HF_TOKEN=os.getenv("HF_TOKEN"),
            HF_MODEL=os.getenv("HF_MODEL", "microsoft/Phi-4"),
            REQUEST_TIMEOUT_SECONDS=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120")),
            MAX_OUTPUT_TOKENS=int(os.getenv("MAX_OUTPUT_TOKENS", "3000")),
            EVIDENCE_FUZZY_THRESHOLD=float(os.getenv("EVIDENCE_FUZZY_THRESHOLD", "0.70")),
            RESULTS_DIRECTORY=os.getenv("RESULTS_DIRECTORY", str(Path(__file__).resolve().parents[1] / "results")),
        )
        try:
            _SETTINGS.RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return _SETTINGS


settings = get_settings()
