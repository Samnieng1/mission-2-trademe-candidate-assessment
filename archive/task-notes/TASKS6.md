Update the existing application so it supports CV-only analysis throughout.

The UI already contains only a CV upload control. Do not add cover-letter inputs, modes, fields or scoring.

1. Remove any remaining cover-letter assumptions from the analysis flow.

The application should always:

- analyse mandatory requirements;
- analyse preferred requirements;
- analyse candidate evidence;
- identify missing or unclear evidence;
- calculate a Profile Fit Score.

Do not:

- assess cover-letter quality;
- return unsupported cover-letter claims;
- return cover-letter feedback;
- award or deduct cover-letter points;
- expect cover-letter fields in the response schema.

2. Update the model response schema for CV-only analysis.

Use this structure:

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
  ]
}

Remove these fields wherever they are no longer needed:

- unsupported_cover_letter_claims;
- cover_letter_feedback;
- cover_letter_score;
- requirements_addressed_in_cover_letter;
- cover_letter_quality.

If the controlled benchmark experiment still uses an older schema, create a separate CV-only demo schema rather than forcing unused cover-letter fields into the demo.

3. Update the shared prompt.

The model should receive only:

- the selected job description;
- the uploaded CV.

Instruct both GPT-5 and Phi-4 to:

- extract only requirements present in the job description;
- separate mandatory and preferred requirements;
- match each requirement to explicit or reasonably supported CV evidence;
- distinguish matched, partially_matched, not_matched and uncertain;
- quote or closely paraphrase candidate evidence;
- not infer qualifications, licences, years of experience or New Zealand work eligibility unless supported by the CV;
- not invent additional requirements;
- return one JSON object, never a list;
- return empty arrays rather than omit fields;
- return JSON only, without markdown fences.

4. Calculate the Profile Fit Score deterministically in Python.

Weighting:

- mandatory requirements: 80 points;
- preferred requirements: 20 points.

Match values:

- matched = 1.0;
- partially_matched = 0.5;
- uncertain = 0.25;
- not_matched = 0.0.

Mandatory score:

sum of mandatory match values
divided by the total number of mandatory benchmark requirements
multiplied by 80.

Preferred score:

sum of preferred match values
divided by the total number of preferred benchmark requirements
multiplied by 20.

Final Profile Fit Score:

mandatory score + preferred score.

Clamp the result between 0 and 100.

The model must not generate the Profile Fit Score.

5. Use benchmark requirement IDs as the scoring source of truth.

Validate that:

- each benchmark requirement appears exactly once in requirement_matches;
- a missing requirement is treated as not_matched and produces a warning;
- duplicate requirement IDs produce a warning;
- extra model-created requirement IDs are ignored;
- invented requirements do not affect the score.

6. Add defensive response normalisation.

Before Pydantic validation:

- unwrap a one-item top-level list containing one object;
- map assessment "met" to status "matched";
- map "partial" or "partially met" to "partially_matched";
- map "missing", "unmet" or "not met" to "not_matched";
- map "unclear" to "uncertain";
- map description to requirement where required;
- if evidence is returned as a list, select or join the strongest evidence into candidate_evidence;
- record whether normalisation occurred.

Do not silently accept arbitrary malformed responses.

7. Update the result display.

Show a Profile Fit section containing:

- mandatory score;
- preferred score;
- total Profile Fit Score;
- mandatory matched count;
- preferred matched count;
- partial matches;
- uncertain matches;
- missing requirements.

Show a Requirement Analysis section containing:

- requirement ID;
- requirement text;
- status;
- candidate evidence;
- reason;
- missing or unclear evidence.

Show Provider Details containing:

- provider;
- model;
- response time;
- schema validity;
- token usage when available;
- estimated cost when configured;
- whether response normalisation occurred.

If Both is selected, display GPT-5 and Phi-4 side by side and compare:

- Profile Fit Score;
- mandatory score;
- preferred score;
- matched requirements;
- partial matches;
- uncertain matches;
- missing requirements;
- response time;
- schema validity;
- token usage;
- estimated cost.

Do not automatically declare a winner.

8. Keep experiment metrics separate from Profile Fit Score.

The controlled AI Model Evaluation section may continue to display:

- precision;
- recall;
- F1;
- evidence-match accuracy;
- JSON validity;
- latency;
- cost;
- consistency.

For an uploaded CV without a manually created candidate-specific benchmark, do not display evidence-match precision, recall, F1 or benchmark accuracy.

The selected job benchmark defines the job requirements, but it does not define the expected candidate matches for a new uploaded CV.

9. Add or update tests for:

- CV-only analysis;
- no cover-letter fields required;
- excellent CV produces a high Profile Fit Score;
- poor CV produces a low Profile Fit Score;
- one-item list response is unwrapped;
- "met" is normalised to "matched";
- evidence lists are normalised correctly;
- missing benchmark requirement;
- duplicate requirement IDs;
- extra model-created requirement ignored;
- all mandatory matched and both preferred partially matched;
- empty CV;
- whitespace-only CV;
- schema validation failure does not crash the app;
- no cover-letter fields appear in the demo output;
- no cover-letter bonus or penalty is applied.

10. Update README documentation.

State clearly that:

- the demo analyses CVs only;
- cover-letter analysis is outside the current scope;
- GPT-5 and Phi-4 extract and match evidence;
- the Profile Fit Score is calculated by deterministic application logic;
- the score is not a hiring probability or hiring decision;
- experiment evaluation metrics and candidate Profile Fit Score are different concepts.

Preserve the existing provider abstraction and reuse shared parsing, scoring and validation logic wherever possible.