from src.prompts import SYSTEM_PROMPT, build_user_message


def test_system_prompt_requires_source_aligned_candidate_evidence():
    lowered = SYSTEM_PROMPT.lower()

    assert "shortest relevant phrase or sentence" in lowered
    assert "do not invent, summarize, or rewrite the evidence" in lowered
    assert "do not merge multiple separate cv snippets" in lowered


def test_user_message_reinforces_verbatim_candidate_evidence_rule():
    payload = build_user_message("CASE1", "Job description", "Candidate CV text")
    lowered = payload["content"].lower()

    assert "quote or closely copy the shortest relevant phrase or sentence from the cv" in lowered
    assert "do not summarize or merge separate snippets" in lowered