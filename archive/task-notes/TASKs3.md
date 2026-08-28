Update the existing Streamlit application to support two clearly separated modes:

📊 AI Model Evaluation
💼 Candidate Application Demo

Do not remove any existing functionality. Refactor the application so both modes reuse the same provider classes, prompt template, schemas and scoring logic wherever appropriate.

# Mode 1 – AI Model Evaluation
This mode is used for the research experiment described in the Technology Evaluation Brief.

Keep all existing functionality including:
predefined fictional benchmark cases
GPT-5
Phi-4
repeated runs
benchmark evaluation
precision
recall
F1
JSON validation
latency
token usage
cost (when configured)
consistency
CSV and JSON export

This mode continues to compare model performance against the manually created benchmark.
No major UI changes are required.

UI of AI Model Evaluation example:
🤖 AI Model Evaluation
Precision: 0.92
Recall: 0.88
F1: 0.90
JSON Valid: ✅
Response Time: 2.1 s
# Mode 2 – Candidate Assessment
Create a new tab called:

💼 Candidate Assessment
This demonstrates the proposed Trade Me Jobs AI Application Fit Coach.

The workflow should simulate a real candidate applying for an existing advertised position.

Do NOT allow uploading a custom job advertisement.

Instead, allow the user to choose one of the existing benchmark jobs.
UI of Candidate Assessemnt example:
👤 Candidate Assessment
Candidate Fit Score: 83%
Mandatory Requirements: 4/5
Preferred Requirements: 2/3
Unsupported Claims: 1
Overall Assessment: Strong Match
## Step 1

Display available positions.

Example:
Available Positions

🖥 Software Developer

🏥 Registered Nurse

🛒 Retail Manager
Each option corresponds to an existing benchmark case already stored under:
data/cases/
## Step 2

After a job is selected, display the complete job advertisement in a read-only expandable panel.

The job advertisement should come directly from the benchmark case.

## Step 3

Allow the user to provide:

Candidate CV

Either

Upload DOCX
Upload PDF
Upload TXT

OR

Paste text

If both are provided, use the uploaded file.
### Cover Letter
Either

Upload DOCX
Upload PDF
Upload TXT

OR

Paste text

If both are provided, use the uploaded file.
## Step 4

Allow provider selection.

Options:

GPT-5
Phi-4
Both
## Step 5
Analyse the uploaded application against the selected benchmark job.

Reuse exactly the same shared prompt already used in the experiment.

The only difference is:

Instead of the benchmark CV and benchmark cover letter,

use

uploaded CV
uploaded cover letter

## Candidate Fit Score

Unlike the benchmark experiment,

this mode SHOULD calculate and display a Candidate Fit Score.

The score must NOT come from the AI model.

The score must be calculated deterministically in Python.

Implement a configurable scoring engine.

Suggested default weighting:
Mandatory requirements
70%

Preferred requirements
20%

Penalty for unsupported claims
10%
Example:
Mandatory
4 of 5 matched

= 56 points

Preferred
2 of 3 matched

= 13 points

Unsupported claims

1

Penalty

−5

Total

64%
The weighting should be configurable from one file.

Do NOT hardcode it throughout the application.
## Dashboard
Display a modern recruiter-style dashboard.

Show:
Candidate Fit Score

83%

★★★★☆

Strong Match

Also display:
Mandatory Requirements

✓ Matched

⚠ Partially Matched

✗ Missing
Preferred Requirements

✓ Matched

⚠ Partially Matched

✗ Missing

Display:

matched evidence
missing evidence
unsupported cover-letter claims
strengths
weaknesses
improvement suggestions

## Provider Comparison

If

Both

is selected,

display two result cards.

Example:
GPT-5

Candidate Score

83%

Latency

2.1 sec

Phi-4

Candidate Score

79%

Latency

1.3 sec

Below them display a comparison table.
Include:

Candidate Fit Score
Mandatory matched
Preferred matched
Unsupported claims
Response time
Token usage
Estimated cost (if configured)

Do NOT automatically declare a winner.

## Important
Because the Candidate Application Demo uses one of the predefined benchmark job descriptions, calculate and display both AI evaluation metrics (Precision, Recall, F1, JSON validity, latency and cost) and candidate assessment metrics (Candidate Fit Score, requirement matching summary, missing evidence and recommendations). Clearly separate these into two sections so users understand that AI evaluation metrics measure model performance against the benchmark, while the Candidate Fit Score measures the uploaded application's alignment with the selected job.

## Privacy
Display the following warning above the upload controls:
Use fictional candidate information only.

Uploaded text may be sent to an external AI provider.

Files are processed only for this session and are not intended for permanent storage.
## File Support

Support

TXT
DOCX
text-based PDF

Maximum upload size: 5 MB
Reject scanned PDFs with a clear message.
## New Python Module
Create: src/candidate_scoring.py
Move all candidate scoring logic into this module.

Provide functions such as:
calculate_candidate_fit_score()

calculate_mandatory_score()

calculate_preferred_score()

calculate_penalty()

generate_summary()

These functions must be deterministic.

They must never call an AI model.

# Code Reuse
Reuse:

provider abstraction
prompt
schemas
document parser
validation
storage

Avoid duplicating logic.
# README
Update the README.

Clearly explain the difference between the two modes.
## AI Model Evaluation
Purpose:

Research experiment comparing GPT-5 and Phi-4.

Produces:

Precision
Recall
F1
Latency
Cost
Consistency

Used in the Technology Evaluation Brief.
## Candidate Application Demo
Purpose:

Demonstrate the proposed Trade Me Jobs AI Application Fit Coach.

Workflow:

Select benchmark job
Upload candidate CV
Upload cover letter
Analyse application
Produce Candidate Fit Score
Display evidence and recommendations

This mode is intended for the presentation and defence.

The Candidate Fit Score is calculated by deterministic application logic rather than by the AI model.

Instead of a generic 83%, give the score an interpretation, for example:
| Score         | Label           |
| ------------- | --------------- |
| **90–100%**   | Excellent Match |
| **75–89%**    | Strong Match    |
| **60–74%**    | Moderate Match  |
| **40–59%**    | Weak Match      |
| **Below 40%** | Poor Match      |
