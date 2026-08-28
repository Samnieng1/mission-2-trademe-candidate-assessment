"""Phi-4 provider adapter using Hugging Face Inference Providers (`InferenceClient`).

This adapter calls the Hugging Face Chat Completions API via `InferenceClient` and
returns a `ProviderResult` compatible with the rest of the application.
"""
from __future__ import annotations

import json
import time
from typing import Optional, Any, Dict, List
import traceback

from ..config import settings
from ..prompts import build_user_message
from ..schemas import ProviderResult, ModelResponse
from .base import ModelProvider, extract_json_payload
from ..normalise import normalise_payload
import logging

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
        "input_tokens_details",
        "output_tokens_details",
    ):
        try:
            if hasattr(obj, key):
                out[key] = getattr(obj, key)
        except Exception:
            pass
    return out

try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None

try:
    from openai import OpenAI as OpenAIRouterClient
except Exception:
    OpenAIRouterClient = None




class Phi4Provider(ModelProvider):
    def __init__(self):
        self.client: Optional[InferenceClient] = None
        self.router_client = None
        if settings.HF_TOKEN:
            try:
                self.client = InferenceClient(api_key=settings.HF_TOKEN)
            except Exception:
                self.client = None
            # Also prepare an OpenAI-compatible router client as a fallback
            if OpenAIRouterClient is not None:
                try:
                    self.router_client = OpenAIRouterClient(
                        base_url="https://router.huggingface.co/v1",
                        api_key=settings.HF_TOKEN,
                    )
                except Exception:
                    self.router_client = None
        self.model = settings.HF_MODEL

    def health(self) -> bool:
        return self.client is not None

    def _extract_text_from_response(self, resp) -> Optional[str]:
        # Try common response shapes returned by InferenceClient
        try:
            if resp is None:
                return None
            # dict-like
            if isinstance(resp, dict):
                if "error" in resp:
                    return None
                if "generated_text" in resp:
                    return resp["generated_text"]
                if "content" in resp:
                    return resp["content"]
                if "outputs" in resp and isinstance(resp["outputs"], list) and resp["outputs"]:
                    out = resp["outputs"][0]
                    if isinstance(out, dict) and "content" in out:
                        return out["content"]
                if "choices" in resp and isinstance(resp["choices"], list) and resp["choices"]:
                    choice = resp["choices"][0]
                    if isinstance(choice, dict):
                        if "message" in choice and isinstance(choice["message"], dict):
                            # some HF formats: {'choices':[{'message':{'content': '...'}}]}
                            return choice["message"].get("content") or choice["message"].get("text")
                        return choice.get("text") or choice.get("message")

            # object-like: try attributes
            if hasattr(resp, "generated_text"):
                return getattr(resp, "generated_text")
            if hasattr(resp, "text"):
                return getattr(resp, "text")
            if hasattr(resp, "content"):
                return getattr(resp, "content")
            # object-like: choices/message structure (InferenceClient returns objects)
            if hasattr(resp, "choices") and getattr(resp, "choices"):
                try:
                    choice = resp.choices[0]
                    if hasattr(choice, "message") and hasattr(choice.message, "content"):
                        return choice.message.content
                    if hasattr(choice, "text"):
                        return choice.text
                    # dict-like inside object
                    if isinstance(choice, dict):
                        if "message" in choice and isinstance(choice["message"], dict):
                            return choice["message"].get("content") or choice["message"].get("text")
                except Exception:
                    pass

            # Fallback to string serialization
            return str(resp)
        except Exception:
            return None

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
            # Normalize candidate evidence into semicolon-separated string
            if value is None:
                return ""
            if isinstance(value, list):
                return "; ".join(str(item) for item in value if item is not None)
            return str(value)

        def normalize_requirements(items: Any) -> List[Dict[str, str]]:
            normalized = []
            if not items:
                return normalized
            # If items is a dict mapping id -> data, convert
            if isinstance(items, dict):
                for rid, rdata in items.items():
                    if not isinstance(rdata, dict):
                        continue
                    normalized.append(
                        {
                            "id": rid,
                            "requirement": rdata.get("requirement") or rdata.get("description") or rdata.get("text") or "",
                            "source_evidence": as_text(rdata.get("evidence") or rdata.get("source") or rdata.get("evidence_job_ad") or ""),
                        }
                    )
                return normalized
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                normalized.append(
                    {
                        "id": item.get("id", ""),
                        "requirement": item.get("requirement") or item.get("text") or item.get("description") or "",
                        "source_evidence": as_text(item.get("source_evidence") or item.get("evidence_job_ad") or item.get("evidence") or ""),
                    }
                )
            return normalized

        requirement_matches = payload.get("requirement_matches")
        if not requirement_matches:
            requirement_matches = []

            def iter_requirements(obj: Any):
                if not obj:
                    return []
                if isinstance(obj, dict):
                    items = []
                    for rid, rdata in obj.items():
                        if isinstance(rdata, dict):
                            entry = dict(rdata)
                            entry["id"] = rid
                            items.append(entry)
                    return items
                if isinstance(obj, list):
                    return obj
                return []

            for item in iter_requirements(payload.get("mandatory_requirements")) + iter_requirements(payload.get("preferred_requirements")):
                if not isinstance(item, dict) or not item.get("id") or not item.get("status"):
                    continue
                # Normalize status and evidence defensively
                raw_status = item.get("status")
                if isinstance(raw_status, list) and len(raw_status) == 1:
                    raw_status = raw_status[0]
                normalized_status = status_map.get(str(raw_status).strip().lower(), item.get("status"))
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
        if unsupported_claims is None:
            unsupported_claims = payload.get("unsupported_claims", [])

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
        res = ProviderResult(provider_name="phi4", model_name=self.model)

        if not self.client and not self.router_client:
            res.error = "Missing HF_TOKEN configuration or both InferenceClient and router client initialization failed"
            res.success = False
            return res

        payload = build_user_message("case", job_description, candidate_profile)
        messages = [
            {"role": "system", "content": payload.get("instructions", "")},
            {"role": "user", "content": payload.get("content", "")},
        ]

        # Try once, then retry one more time on JSON validation failure
        attempts = 0
        last_error = None
        while attempts < 2:
            attempts += 1
            try:
                # Attempt to call common chat completion entrypoints on the client
                resp = None
                try:
                    # Prefer official InferenceClient chat completions if available
                    if self.client is not None:
                        if hasattr(self.client, "chat") and hasattr(self.client.chat, "completions") and hasattr(self.client.chat.completions, "create"):
                            resp = self.client.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                temperature=0,
                                max_tokens=settings.MAX_OUTPUT_TOKENS,
                            )
                        elif hasattr(self.client, "chat_completions"):
                            resp = self.client.chat_completions.create(
                                model=self.model,
                                messages=messages,
                                temperature=0,
                                max_tokens=settings.MAX_OUTPUT_TOKENS,
                            )
                        elif hasattr(self.client, "chat_completion") and callable(self.client.chat_completion):
                            resp = self.client.chat_completion(
                                model=self.model,
                                messages=messages,
                                temperature=0,
                                max_tokens=settings.MAX_OUTPUT_TOKENS,
                            )
                        elif hasattr(self.client, "chat") and hasattr(self.client.chat, "create"):
                            resp = self.client.chat.create(
                                model=self.model,
                                messages=messages,
                                temperature=0,
                                max_tokens=settings.MAX_OUTPUT_TOKENS,
                            )
                        else:
                            # Generic fallback: the InferenceClient may support a `create` helper
                            resp = self.client.create(
                                model=self.model,
                                inputs={"messages": messages},
                                parameters={"temperature": 0, "max_new_tokens": settings.MAX_OUTPUT_TOKENS},
                            )
                    else:
                        resp = None
                except Exception as e:
                    # If InferenceClient fails due to permissions or other issues, try router fallback
                    resp = None
                    infer_exc = e

                res.elapsed_seconds = time.time() - start

                # If the InferenceClient gave no response (e.g., permission error), try the OpenAI-compatible router
                if resp is None and self.router_client is not None:
                    try:
                        # router client follows OpenAI-like interface
                        r = self.router_client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=0,
                            max_tokens=settings.MAX_OUTPUT_TOKENS,
                        )
                        resp = r
                    except Exception as e:
                        # capture router error as message
                        infer_exc = e

                # If we still have no response but captured an inference exception, return a helpful error
                if resp is None and 'infer_exc' in locals() and infer_exc is not None:
                    msg = str(infer_exc)
                    if "does not exist" in msg or "model_not_found" in msg or "The requested model" in msg:
                        res.error = f"Requested model {self.model} not found or inaccessible: {infer_exc}"
                    elif "401" in msg or "Unauthorized" in msg or "Invalid token" in msg:
                        res.error = "Invalid Hugging Face token or authentication failed"
                    elif "403" in msg or "sufficient permissions" in msg:
                        res.error = "HF_TOKEN is valid but lacks Hugging Face Inference Providers permissions"
                    else:
                        res.error = f"Inference client error: {infer_exc}"
                    res.success = False
                    return res

                # Extract text from the returned response
                raw_text = self._extract_text_from_response(resp)
                res.raw_response = raw_text if raw_text is not None else str(resp)

                # Diagnostic logging (avoid logging full CV content)
                try:
                    logger.debug("phi4: raw_text type=%s, length=%d", type(raw_text).__name__, len(raw_text) if raw_text else 0)
                except Exception:
                    logger.debug("phi4: raw_text logging failed")

                # Try to parse JSON directly
                parsed_json = extract_json_payload(raw_text)
                if parsed_json is None and res.raw_response:
                    try:
                        alt = json.loads(res.raw_response)
                        if isinstance(alt, list) and alt and isinstance(alt[0], dict) and "generated_text" in alt[0]:
                            parsed_json = extract_json_payload(alt[0]["generated_text"])
                    except Exception:
                        parsed_json = None

                if parsed_json is None:
                    last_error = "Response did not contain valid JSON"
                    # Try again if we have another attempt
                    continue

                try:
                    # Log parsed_json top-level types to help debug type issues
                    if parsed_json is not None:
                        try:
                            if isinstance(parsed_json, dict):
                                logger.debug("phi4: parsed_json top types: %s", {k: type(v).__name__ for k, v in list(parsed_json.items())[:30]})
                        except Exception:
                            logger.debug("phi4: failed to summarize parsed_json types")
                    # If requirement_matches present, log each status/evidence type
                    if isinstance(parsed_json, dict) and parsed_json.get("requirement_matches"):
                        for i, itm in enumerate(parsed_json.get("requirement_matches") or []):
                            try:
                                st = itm.get("status") if isinstance(itm, dict) else getattr(itm, "status", None)
                                ev = itm.get("candidate_evidence") if isinstance(itm, dict) else getattr(itm, "candidate_evidence", None)
                                logger.debug("phi4: match[%d] status_type=%s status=%s evidence_type=%s", i, type(st).__name__, str(st)[:100], type(ev).__name__)
                            except Exception:
                                pass
                except Exception:
                    pass

                try:
                    # Defensive normalisation step
                    try:
                        normalised, warnings = normalise_payload(parsed_json)
                        if normalised is not None:
                            parsed_json = self._normalize_response_payload(normalised)
                            res.normalization_applied = True
                            res.normalization_warnings = warnings
                        else:
                            parsed_json = self._normalize_response_payload(parsed_json)
                            if warnings:
                                res.normalization_warnings = warnings
                    except Exception:
                        parsed_json = self._normalize_response_payload(parsed_json)

                    validated = ModelResponse.parse_obj(parsed_json)
                    res.parsed_response = validated
                    res.validation_status = True
                    res.success = True

                    # Try to extract usage/cost info if present
                    try:
                        def _extract_tokens_from_usage(obj: Any) -> (int, int):
                            if not obj:
                                return None, None
                            usage_dict = _usage_to_dict(obj)
                            in_t = usage_dict.get("input_tokens") or usage_dict.get("prompt_tokens") or usage_dict.get("prompt_token_count")
                            out_t = usage_dict.get("output_tokens") or usage_dict.get("completion_tokens") or usage_dict.get("generated_tokens")
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
                        if isinstance(resp, dict) and "usage" in resp:
                            usage = resp["usage"]
                        elif hasattr(resp, "usage"):
                            usage = getattr(resp, "usage")

                        in_t, out_t = None, None
                        if usage:
                            in_t, out_t = _extract_tokens_from_usage(usage)
                            usage_dict = _usage_to_dict(usage)
                            if in_t is None and isinstance(usage_dict, dict):
                                for v in usage_dict.values():
                                    if isinstance(v, dict):
                                        in_t, out_t = _extract_tokens_from_usage(v)
                                        if in_t is not None or out_t is not None:
                                            break
                            logger.debug("phi4: usage extracted usage_keys=%s input_tokens=%s output_tokens=%s", list(usage_dict.keys()) if isinstance(usage_dict, dict) else [], in_t, out_t)
                        else:
                            # Try OpenAI-like nested choices usage
                            try:
                                if isinstance(resp, dict) and resp.get("choices"):
                                    ch = resp["choices"][0]
                                    if isinstance(ch, dict) and "usage" in ch:
                                        in_t, out_t = _extract_tokens_from_usage(ch["usage"])
                                        logger.debug("phi4: usage extracted from choice input_tokens=%s output_tokens=%s", in_t, out_t)
                            except Exception:
                                in_t, out_t = None, None

                        if in_t is None and out_t is None:
                            logger.debug("phi4: no token usage returned by response")

                        if in_t is not None:
                            res.input_tokens = in_t
                        if out_t is not None:
                            res.output_tokens = out_t
                    except Exception:
                        pass

                    return res
                except Exception as e:
                    # Add diagnostic summary of parsed_json top-level types to aid debugging
                    def _summarize_types(o: Any) -> Any:
                        try:
                            if isinstance(o, dict):
                                return {k: type(v).__name__ for k, v in list(o.items())[:20]}
                            return type(o).__name__
                        except Exception:
                            return str(type(o))

                    types_summary = _summarize_types(parsed_json)
                    last_error = f"Validation error: {e}; parsed_json_types={types_summary}; normalized_payload={parsed_json}"
                    # on validation failure, retry once
                    continue

            except Exception as e:
                # Map common errors to user-friendly messages
                msg = str(e)
                # Detect model-not-found / inaccessible model cases
                if "does not exist" in msg or "model_not_found" in msg or "The requested model" in msg:
                    res.error = f"Requested model {self.model} not found or inaccessible: {e}"
                    res.success = False
                    return res
                if "401" in msg or "Unauthorized" in msg or "Invalid token" in msg:
                    res.error = "Invalid Hugging Face token or authentication failed"
                elif "403" in msg or "sufficient permissions" in msg:
                    res.error = "HF_TOKEN is valid but lacks Hugging Face Inference Providers permissions"
                elif "rate" in msg.lower():
                    res.error = "Rate limited by Hugging Face Inference API"
                else:
                    # Capture full traceback for debugging
                    tb = traceback.format_exc()
                    res.error = f"API request failed: {e}; repr={repr(e)}; traceback={tb}"
                res.success = False
                return res

        # If we exit the loop without success, return a structured error
        res.elapsed_seconds = time.time() - start
        res.success = False
        res.error = last_error or "Unknown error from Hugging Face Inference API"
        return res

