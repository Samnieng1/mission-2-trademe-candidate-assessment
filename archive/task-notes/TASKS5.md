Fix the model response parsing and Candidate Fit Score calculation.

Current issue:

The model is successfully identifying candidate matches, but it sometimes
returns a list instead of a top-level JSON object. In a valid response, it
returns fields such as:

- description
- evidence
- assessment: "met"

The application expects:

- requirement
- source_evidence
- requirement_matches
- candidate_evidence
- status: "matched"

As a result, Pydantic validation fails in two of three runs, evidence-match
accuracy is zero, and Candidate Fit Score is incorrectly calculated as 0.0.

Make the following changes.

1. Enforce one canonical ModelResponse schema:

{
  "case_id": "string",
  "mandatory_requirements": [
    {
      "id": "M1",
      "requirement": "string",
      "source_evidence": "string"
    }
  ],
  "preferred_requirements": [
    {
      "id": "P1",
      "requirement": "string",
      "source_evidence": "string"
    }
  ],
  "requirement_matches": [
    {
      "requirement_id": "M1",
      "status": "matched | partially_matched | not_matched | uncertain",
      "candidate_evidence": "string",
      "reason": "string"
    }
  ],
  "missing_evidence": [
    {
      "requirement_id": "M1",
      "reason": "string"
    }
  ],
  "unsupported_cover_letter_claims": [
    {
      "claim": "string",
      "reason": "string"
    }
  ],
  "cover_letter_feedback": {
    "strengths": [],
    "weaknesses": [],
    "suggestions": []
  }
}

2. Update the shared prompt so both GPT-5 and Phi-4 must:

- return one JSON object, never a list;
- use exactly the field names defined by ModelResponse;
- use only these match statuses:
  matched, partially_matched, not_matched, uncertain;
- never use "met", "partial", "yes" or "no";
- include one requirement_matches record for every mandatory and preferred
  benchmark requirement;
- use the benchmark requirement ID supplied by the application;
- extract only requirements present in the job advertisement;
- not create extra requirements such as communication or trade-off reasoning
  unless they explicitly appear in the job advertisement;
- return empty arrays rather than omitting fields;
- return only JSON without markdown fences.

3. Where supported, use provider-native structured output or JSON schema.

4. Add a defensive normalisation layer before Pydantic validation:

- if assessment == "met", map to "matched";
- if assessment == "partial", map to "partially_matched";
- if assessment == "not met" or "missing", map to "not_matched";
- if evidence is a list, join it into candidate_evidence or retain the
  strongest exact evidence item;
- map unsupported_claims to unsupported_cover_letter_claims;
- if the top-level response is a one-item list containing an object, unwrap it;
- do not silently accept arbitrary malformed structures;
- record that normalisation occurred.

5. Fix Candidate Fit Score so it reads from requirement_matches.

Suggested scoring:

- matched = 1.0
- partially_matched = 0.5
- uncertain = 0.25
- not_matched = 0.0

Mandatory component:
sum of mandatory match values / number of mandatory requirements * 70

Preferred component:
sum of preferred match values / number of preferred requirements * 20

Unsupported-claim component:
start with 10 points and subtract 5 points per unsupported claim, with a
minimum of 0.

Total score:
mandatory component + preferred component + unsupported-claim component

Clamp the final result between 0 and 100.

6. Validate that every benchmark requirement ID appears exactly once in
requirement_matches.

If a requirement match is absent, treat it as not_matched and record a warning.

Ignore model-created requirement IDs that do not exist in the selected
benchmark.

7. Add unit tests for:

- model response returned as a one-item list;
- assessment "met" normalised to "matched";
- evidence list normalised correctly;
- missing requirement match;
- extra P3 requirement ignored when benchmark contains only P1 and P2;
- excellent candidate produces a high score;
- all mandatory matched and both preferred partially matched produces the
  expected deterministic score;
- invalid top-level list with multiple objects remains a validation error.

8. Display separately:

AI model quality metrics:
- precision
- recall
- F1
- evidence-match accuracy
- schema validity

Candidate assessment:
- Candidate Fit Score
- mandatory match score
- preferred match score
- unsupported-claim component

Do not treat Candidate Fit Score as model accuracy.