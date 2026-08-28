Add two modes near the top of the page:

1. Benchmark Experiment

Keep your current interface for the formal comparison:

Three predefined fictional cases
Same inputs for GPT and Phi-4
Human benchmark
Repeated runs
Precision, recall and F1
Latency, consistency and JSON validity

Use results from this mode in your assignment.
2. Custom Application Demo

Add a separate tab for demonstrating how the proposed feature might work:

Paste or upload a job advertisement
Upload a fictional CV
Upload a fictional cover letter
Select GPT-5, Phi-4 or both
Run the analysis
Display matched requirements, missing evidence, unsupported claims and cover-letter feedback

Do not calculate benchmark accuracy in this mode because there is no researcher-created reference answer.

A suitable layout would be:
[ Benchmark Experiment ] [ Custom Application Demo ] [Provider Comparison]

# custom-demo interface
Custom Application Demo

Job advertisement
[Paste text here]
or
[Upload TXT/PDF/DOCX]

Candidate CV
[Upload TXT/PDF/DOCX]
or
[Paste CV text]

Cover letter
[Upload TXT/PDF/DOCX]
or
[Paste cover-letter text]

Provider
[OpenAI] [Phi-4] [Both]

[Analyse application]

# Then display results under these headings:
Mandatory requirements
Preferred requirements
Matched candidate evidence
Partially matched requirements
Missing evidence
Unsupported cover-letter claims
Cover-letter strengths
Cover-letter weaknesses
Suggested improvements

Support:

.txt
.docx
text-based .pdf
Maximum file size: 5 MB
Do not store uploaded files permanently. Extract the text in memory and send only the extracted content to the selected model.

the model should receive a natural job advertisement:
Example, Applicants must hold a current Registered Nurse practising certificate and
have at least two years of recent clinical nursing experience. Experience
administering medication and IV therapy is essential...

# Prompt
Update the existing Streamlit experiment application by adding two clearly
separated modes using tabs:

1. "Benchmark Experiment"
2. "Custom Application Demo"
3. Provider Comparision

Do not remove or change the existing benchmark experiment functionality.

The Benchmark Experiment tab must continue to use the predefined fictional
test cases, researcher-created benchmarks, repeated runs, deterministic
metrics, result storage and GPT-versus-Phi-4 comparison.

Add a new Custom Application Demo tab.

The Custom Application Demo must allow the user to:

- paste a job advertisement into a text area;
- optionally upload a job advertisement as TXT, DOCX or text-based PDF;
- paste a candidate CV into a text area;
- optionally upload a candidate CV as TXT, DOCX or text-based PDF;
- paste a cover letter into a text area;
- optionally upload a cover letter as TXT, DOCX or text-based PDF;
- select OpenAI, Phi-4 or both providers;
- run the same shared analysis prompt used by the benchmark experiment.

Implement document text extraction in a new file:

src/document_parser.py

Use:

- standard Python file decoding for TXT;
- python-docx for DOCX;
- pypdf for PDF.

Do not use OCR. If a PDF contains no extractable text, display a clear message
that scanned or image-only PDFs are not supported.

Limit uploaded files to 5 MB.

For each document, allow either pasted text or an uploaded file. If both are
provided, prefer the uploaded file and display a message explaining which
source was used.

Before analysis, validate that:

- the job advertisement is not empty;
- the CV is not empty;
- the cover letter is not empty.

Display a privacy warning:

"Use fictional candidate information only. Submitted text may be processed by
an external AI provider. Uploaded files are parsed for this session and are
not intended for permanent storage."

The custom mode must display:

- mandatory requirements;
- preferred requirements;
- matched evidence;
- partially matched evidence;
- missing evidence;
- unsupported cover-letter claims;
- cover-letter strengths;
- cover-letter weaknesses;
- improvement suggestions;
- provider name;
- model name;
- response time;
- token usage when available;
- schema validity;
- raw JSON inside an expander.

Do not display precision, recall, F1 or benchmark accuracy for custom uploads,
because no human benchmark exists.

Do not calculate an overall candidate match score.
Do not predict hiring probability.
Do not rank the candidate.
Do not make decisions based on protected personal characteristics.

When both providers are selected, display their outputs side by side where
practical, followed by a comparison table containing:

- response time;
- schema validity;
- number of mandatory requirements extracted;
- number of preferred requirements extracted;
- number of unsupported claims detected;
- token usage when available;
- cost when configured.

Do not automatically declare a winning model.

Add the required packages to requirements.txt:

python-docx
pypdf

Add unit tests for:

- TXT extraction;
- DOCX extraction;
- text-based PDF extraction;
- empty PDF handling;
- unsupported file types;
- files larger than 5 MB;
- input validation.

Mock provider calls in all UI and unit tests.

Also update the README to explain the difference between the controlled
benchmark mode and the custom demonstration mode. State clearly that only the
benchmark mode should be used for the quantitative results in the Technology
Evaluation Brief.