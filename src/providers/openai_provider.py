"""OpenAI provider adapter using the official `openai` Python SDK (Responses API).

This adapter builds a request from the shared prompts and returns a `ProviderResult`.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from openai import OpenAI

from ..config import settings
from ..prompts import build_user_message
from ..schemas import ProviderResult, ModelResponse
from .base import ModelProvider, extract_json_payload
from ..normalise import normalise_payload
import logging
import traceback

logger = logging.getLogger(__name__)


def _usage_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        try:
            dumped = obj.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    out: Dict[str, Any] = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "generated_tokens",
        "prompt_token_count",
        "promptTokens",
        "completionTokens",
        "input_tokens_details",
        "output_tokens_details",
    ):
        try:
            if hasattr(obj, key):
                out[key] = getattr(obj, key)
        except Exception:
            pass
    return out


class OpenAIProvider(ModelProvider):
    def __init__(self):
        self.client = None
        if settings.OPENAI_API_KEY:
            try:
                self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            except Exception:
                self.client = None

    def health(self) -> bool:
        return self.client is not None

    def _normalize_response_payload(self, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload

        status_map = {
            "met": "matched",
            "matched": "matched",
            "partial": "partially_matched",
            "partially_met": "partially_matched",
            "partially_matched": "partially_matched",
            "not_met": "not_matched",
            "not_matched": "not_matched",
            "missing": "not_matched",
            "uncertain": "uncertain",
        }

        def as_text(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, list):
                return "; ".join(str(item) for item in value if item is not None)
            return str(value)

        def normalize_requirements(items: Any) -> List[Dict[str, str]]:
            normalized = []
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                normalized.append(
                    {
                        "id": item.get("id", ""),
                        "requirement": item.get("requirement") or item.get("text") or "",
                        "source_evidence": as_text(item.get("source_evidence") or item.get("evidence_job_ad") or item.get("evidence") or ""),
                    }
                )
            return normalized

        requirement_matches = payload.get("requirement_matches")
        if not requirement_matches:
            requirement_matches = []
            for item in (payload.get("mandatory_requirements") or []) + (payload.get("preferred_requirements") or []):
                if not isinstance(item, dict) or not item.get("id") or not item.get("status"):
                    continue
                normalized_status = status_map.get(str(item.get("status")).strip().lower(), item.get("status"))
                candidate_evidence = as_text(item.get("candidate_evidence") or item.get("evidence_candidate") or item.get("evidence")) or None
                requirement_matches.append(
                    {
                        "requirement_id": item.get("id"),
                        "status": normalized_status,
                        "candidate_evidence": candidate_evidence,
                        "reason": item.get("reason"),
                    }
                )

        # Map alternate field names into expected keys
        unsupported_claims = payload.get("unsupported_cover_letter_claims")
        if unsupported_claims is None:
            unsupported_claims = payload.get("cover_letter_discrepancies", [])

        # Handle alternate 'missing_mandatory' -> missing_evidence
        missing_evidence = []
        if "missing_evidence" in payload and isinstance(payload["missing_evidence"], list):
            missing_evidence = payload["missing_evidence"]
        elif "missing_mandatory" in payload and isinstance(payload["missing_mandatory"], list):
            for mid in payload["missing_mandatory"]:
                missing_evidence.append({"requirement_id": mid, "reason": ""})

        normalized_claims = []
        for item in unsupported_claims or []:
            if isinstance(item, dict):
                normalized_claims.append(
                    {
                        "claim": item.get("claim") or item.get("statement") or "",
                        "reason": item.get("reason") or item.get("explanation") or "",
                    }
                )

        feedback = payload.get("cover_letter_feedback", {})
        if isinstance(feedback, list):
            feedback = {"suggestions": [str(item) for item in feedback]}
        elif not isinstance(feedback, dict):
            feedback = {}

        # If the model returned a `requirements` dict structure, try to map it
        if "requirements" in payload and isinstance(payload["requirements"], dict):
            reqs = payload["requirements"]
            mand = reqs.get("mandatory") or reqs.get("must") or {}
            pref = reqs.get("preferred") or reqs.get("optional") or {}
            # normalize when these are dicts mapping id -> data
            if isinstance(mand, dict):
                mand_items = []
                for rid, rdata in mand.items():
                    if isinstance(rdata, dict):
                        mand_items.append({
                            "id": rid,
                            "requirement": rdata.get("requirement") or rdata.get("text") or "",
                            "source_evidence": as_text(rdata.get("evidence") or rdata.get("source")),
                        })
                mandatory_normalized = mand_items
            else:
                mandatory_normalized = normalize_requirements(mand)

            if isinstance(pref, dict):
                pref_items = []
                for rid, rdata in pref.items():
                    if isinstance(rdata, dict):
                        pref_items.append({
                            "id": rid,
                            "requirement": rdata.get("requirement") or rdata.get("text") or "",
                            "source_evidence": as_text(rdata.get("evidence") or rdata.get("source")),
                        })
                preferred_normalized = pref_items
            else:
                preferred_normalized = normalize_requirements(pref)

        else:
            mandatory_normalized = normalize_requirements(payload.get("mandatory_requirements"))
            preferred_normalized = normalize_requirements(payload.get("preferred_requirements"))

        normalized = {
            "mandatory_requirements": mandatory_normalized,
            "preferred_requirements": preferred_normalized,
            "requirement_matches": requirement_matches,
            "missing_evidence": missing_evidence or payload.get("missing_evidence", []),
        }
        return normalized

    def analyse(self, job_description: str, candidate_profile: str) -> ProviderResult:
        start = time.time()
        res = ProviderResult(provider_name="openai", model_name=settings.OPENAI_MODEL)

        if not self.client:
            res.error = "Missing OpenAI API key or client initialization failed"
            res.success = False
            return res

        payload = build_user_message("case", job_description, candidate_profile)

        try:
            response = self.client.responses.create(
                model=settings.OPENAI_MODEL,
                instructions=payload["instructions"],
                input=payload["content"],
                max_output_tokens=settings.MAX_OUTPUT_TOKENS,
            )
            # Capture raw text for downstream parsing
            raw_text = response.output_text if hasattr(response, "output_text") else str(response)

            # Try to extract token usage if the SDK returns it
            try:
                def _extract_tokens_from_usage(obj: Any) -> (int, int):
                    if not obj:
                        return None, None
                    usage_dict = _usage_to_dict(obj)
                    in_t = usage_dict.get("input_tokens") or usage_dict.get("prompt_tokens") or usage_dict.get("prompt_token_count") or usage_dict.get("promptTokens")
                    out_t = usage_dict.get("output_tokens") or usage_dict.get("completion_tokens") or usage_dict.get("generated_tokens") or usage_dict.get("completionTokens")
                    try:
                        in_val = int(in_t) if in_t is not None else None
                    except Exception:
                        in_val = None
                    try:
                        out_val = int(out_t) if out_t is not None else None
                    except Exception:
                        out_val = None
                    return in_val, out_val

                usage = None
                if isinstance(response, dict) and "usage" in response:
                    usage = response["usage"]
                elif hasattr(response, "usage"):
                    usage = getattr(response, "usage")
                elif hasattr(response, "meta") and isinstance(getattr(response, "meta"), dict) and "usage" in response.meta:
                    usage = response.meta["usage"]

                if usage:
                    in_t, out_t = _extract_tokens_from_usage(usage)
                    if in_t is None or out_t is None:
                        # try nested patterns
                        usage_dict = _usage_to_dict(usage)
                        if isinstance(usage_dict, dict):
                            for v in usage_dict.values():
                                if isinstance(v, dict):
                                    in_t, out_t = _extract_tokens_from_usage(v)
                                    if in_t is not None or out_t is not None:
                                        break
                    logger.debug("openai: usage extracted usage_keys=%s input_tokens=%s output_tokens=%s", list(usage_dict.keys()) if isinstance(usage_dict, dict) else [], in_t, out_t)
                else:
                    # some SDKs provide usage under response['choices'][0]['usage'] or similar
                    try:
                        if isinstance(response, dict) and response.get("choices"):
                            ch = response["choices"][0]
                            if isinstance(ch, dict) and "usage" in ch:
                                in_t, out_t = _extract_tokens_from_usage(ch["usage"])
                                logger.debug("openai: usage extracted from choice input_tokens=%s output_tokens=%s", in_t, out_t)
                    except Exception:
                        in_t, out_t = None, None
                if 'in_t' in locals() and 'out_t' in locals() and in_t is None and out_t is None:
                    logger.debug("openai: no token usage returned by response")

                # Set on result if found
                if 'in_t' in locals() and in_t is not None:
                    res.input_tokens = in_t
                if 'out_t' in locals() and out_t is not None:
                    res.output_tokens = out_t
            except Exception:
                # Non-fatal: token usage is optional
                pass
            res.raw_response = raw_text
            res.elapsed_seconds = time.time() - start
            # Attempt to parse JSON from the output
            try:
                parsed_json = extract_json_payload(raw_text)
                if parsed_json is None:
                    raise json.JSONDecodeError("No JSON payload found", raw_text or "", 0)

                # Defensive normalisation step
                try:
                    normalised, warnings = normalise_payload(parsed_json)
                    if normalised is not None:
                        parsed_json = self._normalize_response_payload(normalised)
                        res.normalization_applied = True
                        res.normalization_warnings = warnings
                    else:
                        # If normaliser couldn't handle payload, fall back to provider-specific normalisation
                        parsed_json = self._normalize_response_payload(parsed_json)
                        if warnings:
                            res.normalization_warnings = warnings
                except Exception:
                    parsed_json = self._normalize_response_payload(parsed_json)

                # Validate using ModelResponse
                try:
                    validated = ModelResponse.parse_obj(parsed_json)
                    res.parsed_response = validated
                    res.validation_status = True
                    res.success = True
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.debug("openai: validation failed: %s", tb)
                    res.error = f"Validation error: {e}; traceback={tb}"
                    res.success = False
                    res.validation_status = False
                    # don't raise; return the ProviderResult with diagnostics
            except json.JSONDecodeError:
                res.error = "Response did not contain valid JSON"
                res.success = False
            except Exception as e:
                tb = traceback.format_exc()
                logger.debug("openai: unexpected error during parsing/validation: %s", tb)
                res.error = f"Validation error: {e}; traceback={tb}"
                res.success = False
        except Exception as e:
            res.error = f"API request failed: {e}"
            res.success = False

        return res

