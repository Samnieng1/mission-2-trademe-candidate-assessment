"""Streamlit UI for the Mission 2 candidate-fit demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import re

import pandas as pd
import streamlit as st

from src.config import settings
from src.candidate_scoring import build_candidate_fit_breakdown
from src.document_parser import parse_uploaded_file
from src.validation import build_assessment_audit_export, run_validated_analysis


st.set_page_config(page_title="TradeMe AI Candidate Fit Demo", layout="wide")


def _score_label_and_stars(pct: float) -> tuple[str, str]:
    if pct >= 90:
        label = "Excellent Match"
    elif pct >= 75:
        label = "Strong Match"
    elif pct >= 60:
        label = "Moderate Match"
    elif pct >= 40:
        label = "Weak Match"
    else:
        label = "Poor Match"

    # 5-star rating
    stars = int(round((pct / 100.0) * 5))
    star_str = "★" * stars + "☆" * (5 - stars)
    return label, star_str


def _clean_job_text(txt: str) -> str:
    if not txt:
        return txt
    return re.sub(r"\s*\([MP]\d+\)", "", txt)


@st.cache_data
def list_cases() -> List[Dict]:
    base = Path("data/cases")
    out = []
    for p in sorted(base.glob("*.json")):
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            out.append(data)
    return out


cases = list_cases()

case_options = {f"{c['case_id']} — {c['job_title']}": c for c in cases}


st.title("Trade Me AI Candidate Fit Demo")
st.markdown("Upload a CV, run Phi-4 analysis, and only use GPT-5 when validation requires review.")
st.warning("Use fictional candidate information only. Uploaded text may be sent to external AI providers. Files are processed for this session and are not intended for permanent storage.")

with st.expander("How the analysis works", expanded=False):
    st.markdown(
        "1. The selected job description and uploaded CV are sent to Phi-4.\n"
        "2. The structured response is validated against the shared schema and evidence-grounding rules.\n"
        "3. If Phi-4 is invalid or a mandatory requirement needs review, GPT-5 is used internally.\n"
        "4. The final Candidate Fit Score is calculated deterministically in Python using 80% mandatory and 20% preferred weighting."
    )

st.subheader("Candidate Assessment")
selected_label = st.selectbox("1. Select job", list(case_options.keys()))
selected_case = case_options[selected_label]

with st.expander("Job advertisement", expanded=True):
    show_ids = st.checkbox("Show internal requirement IDs", value=False)
    text_to_show = selected_case["job_description"] if show_ids else _clean_job_text(selected_case["job_description"])
    st.markdown(text_to_show)

st.markdown("**2. Upload or paste CV**")
uploaded_cv = st.file_uploader("Upload CV (TXT, DOCX, PDF)", type=["txt", "pdf", "docx"], key="cv_upload")
paste_cv = st.text_area("Or paste CV text")

cv_text, cv_err = parse_uploaded_file(uploaded_cv)
if cv_err:
    st.error(f"CV upload: {cv_err}")

final_cv = cv_text if cv_text else paste_cv
analyse_button = st.button("3. Analyse", type="primary")

if analyse_button:
    if not final_cv.strip():
        st.error("No candidate CV provided. Please upload a file or paste the CV text.")
    else:
        with st.spinner("Running candidate analysis..."):
            st.session_state["analysis_result"] = run_validated_analysis(
                selected_case["job_description"],
                final_cv,
            )
            st.session_state["analysis_case"] = selected_case
            st.session_state["analysis_cv"] = final_cv


def _render_breakdown_table(title: str, counts: Dict[str, int], total: int, component: float):
    st.markdown(f"**{title}**")
    st.write(f"Score contribution: {component:.2f} / {80.0 if title == 'Mandatory' else 20.0}")
    st.dataframe(
        pd.DataFrame(
            [
                {"status": "matched", "count": counts.get("matched", 0)},
                {"status": "partially_matched", "count": counts.get("partially_matched", 0)},
                {"status": "uncertain", "count": counts.get("uncertain", 0)},
                {"status": "not_matched", "count": counts.get("not_matched", 0)},
                {"status": "total_requirements", "count": total},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


def _build_audit_export_payload(analysis_result, selected_case: Dict, final_result) -> Dict:
    score_breakdown = (
        build_candidate_fit_breakdown(final_result.parsed_response)
        if getattr(final_result, "parsed_response", None)
        else {}
    )
    return build_assessment_audit_export(analysis_result, selected_case, score_breakdown)


analysis_result = st.session_state.get("analysis_result")
if analysis_result is not None:
    final_result = analysis_result.final_result
    selected_case = st.session_state.get("analysis_case", selected_case)
    export_obj = _build_audit_export_payload(analysis_result, selected_case, final_result)

    if not final_result.success or not final_result.parsed_response:
        st.error(final_result.error or "Analysis could not produce a validated result.")
        with st.expander("Diagnostics", expanded=False):
            st.write({
                "phi4_attempts": analysis_result.phi4_attempts,
                "provider_sequence": analysis_result.provider_sequence,
                "review_reasons": analysis_result.review_reasons,
                "primary_error": getattr(analysis_result.primary_result, "error", None),
                "review_error": getattr(analysis_result.review_result, "error", None) if analysis_result.review_result else None,
            })
        with st.expander("Internal audit trace", expanded=False):
            st.json(export_obj)
        st.download_button(
            "Download assessment JSON",
            json.dumps(export_obj, indent=2).encode("utf-8"),
            file_name=f"assessment_{selected_case.get('case_id')}.json",
            mime="application/json",
        )
    else:
        breakdown = build_candidate_fit_breakdown(final_result.parsed_response)
        label, stars = _score_label_and_stars(breakdown["total_percentage"])

        if analysis_result.gpt5_used:
            st.info("GPT-5 review was used because the initial Phi-4 result required validation review.")
        else:
            st.success("Phi-4 result passed validation without GPT-5 review.")

        corrected_issues = []
        unresolved_mandatory_uncertainty = []
        if analysis_result.validation_outcome is not None:
            corrected_issues = [
                issue.message
                for issue in analysis_result.validation_outcome.issues
                if issue.severity == "corrected"
            ]
            unresolved_mandatory_uncertainty = [
                issue.message
                for issue in analysis_result.validation_outcome.issues
                if issue.code == "mandatory_uncertain"
            ]
        if corrected_issues:
            st.info("Validation applied conservative corrections before scoring.")
        if unresolved_mandatory_uncertainty:
            st.warning("GPT-5 review still found unresolved mandatory requirements. The result is shown with those items marked as uncertain.")

        st.markdown("**4. Candidate Fit Score**")
        st.write(f"Overall score: {breakdown['total_percentage']:.1f} — {label} {stars}")

        col1, col2 = st.columns(2)
        with col1:
            _render_breakdown_table(
                "Mandatory",
                breakdown["mandatory_counts"],
                breakdown["mandatory_total"],
                breakdown["mandatory_component"],
            )
        with col2:
            _render_breakdown_table(
                "Preferred",
                breakdown["preferred_counts"],
                breakdown["preferred_total"],
                breakdown["preferred_component"],
            )

        requirement_map = {
            requirement.id: {"requirement": requirement.requirement, "category": "mandatory"}
            for requirement in final_result.parsed_response.mandatory_requirements
        }
        requirement_map.update(
            {
                requirement.id: {"requirement": requirement.requirement, "category": "preferred"}
                for requirement in final_result.parsed_response.preferred_requirements
            }
        )
        requirement_rows = []
        for match in final_result.parsed_response.requirement_matches:
            meta = requirement_map.get(match.requirement_id, {"requirement": "", "category": "unknown"})
            requirement_rows.append(
                {
                    "requirement_id": match.requirement_id,
                    "category": meta["category"],
                    "requirement": meta["requirement"],
                    "status": getattr(match.status, "value", match.status),
                    "candidate_evidence": match.candidate_evidence or "",
                    "reason": match.reason or "",
                }
            )

        st.markdown("**5. Requirement assessment**")
        st.dataframe(pd.DataFrame(requirement_rows), use_container_width=True, hide_index=True)

        st.markdown("**6. Missing evidence**")
        if final_result.parsed_response.missing_evidence:
            st.dataframe(
                pd.DataFrame([item.dict() for item in final_result.parsed_response.missing_evidence]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("No missing evidence items were returned.")

        with st.expander("Provider details", expanded=False):
            st.write(
                {
                    "primary_provider": analysis_result.primary_result.provider_name,
                    "final_provider": final_result.provider_name,
                    "phi4_attempts": analysis_result.phi4_attempts,
                    "gpt5_review_used": analysis_result.gpt5_used,
                    "schema_valid": final_result.validation_status,
                    "elapsed_seconds": final_result.elapsed_seconds,
                    "input_tokens": final_result.input_tokens,
                    "output_tokens": final_result.output_tokens,
                    "estimated_cost": final_result.estimated_cost,
                    "review_reasons": analysis_result.review_reasons,
                    "corrected_issues": corrected_issues,
                }
            )

        with st.expander("Internal audit trace", expanded=False):
            st.json(export_obj)

        st.download_button(
            "Download assessment JSON",
            json.dumps(export_obj, indent=2).encode("utf-8"),
            file_name=f"assessment_{selected_case.get('case_id')}.json",
            mime="application/json",
        )