# Design Choices behind MedBridge

### 1. How will relevant information be retrieved efficiently?

Two-stage retrieval. First, the user query gets encoded into a 384-dim vector using `all-MiniLM-L6-v2` and searched against a FAISS `IndexFlatIP` index, this is a brute-force cosine similarity search that returns the top 20 candidate chunks in milliseconds (the corpus is under 100k chunks, so exact search is plenty fast). Then a cross-encoder (`ms-marco-MiniLM-L-6-v2`) rescores those 20 candidates by looking at the full (query, chunk) pair jointly, and the top 5 are kept. The bi-encoder is fast but approximate; the cross-encoder is slow but precise. Running the cross-encoder on only 20 candidates instead of the whole corpus makes this tractable.

Both models are loaded once and cached in module-level globals in `retrieval.py`, so no re-initialization is needed per query.

### 2. How will answers remain grounded in evidence?

The system prompt in `generate.py` explicitly tells the model: "Only use information from the provided evidence. If evidence is insufficient, say so. Do not guess." The retrieved evidence chunks are injected directly into the prompt as numbered references `[1]`, `[2]`, etc. The model is forced to output structured JSON via Groq's `response_format={"type": "json_object"}`, which includes an `answer` field that must reference evidence by index, a `citations` array mapping claims to source indices, and a `limitations` field where the model must acknowledge gaps. Temperature is set to 0.1 to minimize creative drift.

If no evidence is retrieved at all, the system short-circuits with a hardcoded "insufficient evidence" response, ensuring that the generation part of the pipeline never runs.

### 3. How will hallucinations be prevented?

Multiple layers:

- **Retrieval gating**: If FAISS returns nothing relevant (or reranker scores are all negative), the system returns a canned "couldn't find enough evidence" response without calling the LLM at all.
- **Grounded prompt**: The system prompt forbids fabrication and requires the model to cite evidence indices for every claim.
- **Structured output**: JSON mode forces the model to produce `citations`, a list that maps each claim to a specific evidence chunk index. This makes unsupported claims structurally visible.
- **Low temperature (0.1)**: Reduces sampling randomness, keeping the model close to the provided context.
- **Post-generation safety check**: `safety.check_response()` scans the output for prescriptive language patterns that would indicate the model overstepped the evidence.
- **Confidence scoring**: If the model makes claims without citations or with weak retrieval scores, `confidence.py` flags this as LOW confidence, signaling to the user that the answer may not be well-supported.

### 4. How will citations be generated?

The LLM is prompted to return a JSON `citations` array where each entry contains a `claim` (the specific statement being cited) and a `source_idx` (which evidence chunk backs it up). Evidence chunks are numbered `[1]` through `[5]` in the prompt, and each chunk carries the metadata `source_name`, `source_url`, `focus_area`, `doc_type` which was attached during ingestion.

In the Streamlit UI, citations render as cards showing the source name, a link to the original source URL, and the claim text. The user can also expand "Retrieved Evidence Chunks" to see the raw chunks with their reranker scores, so they can verify the citation mapping themselves.

### 5. How will confidence be estimated?

`confidence.py` computes a weighted score from three signals:

- **Retrieval quality (45% weight)**: Average reranker score from the cross-encoder, normalized to 0-1. If the best chunk's reranker score is negative, confidence is capped at 0.3 regardless of other signals.
- **Source diversity (30% weight)**: Number of distinct sources in the top evidence. A hypertension query pulling from MedQuAD + WHO + NICE + CDC scores higher than one pulling from MedQuAD alone.
- **Citation coverage (25% weight)**: How many citation entries the LLM produced. More specific citations would mean more verifiable, which leads to higher confidence.

The final score maps to HIGH (≥0.7), MEDIUM (≥0.4), or LOW (<0.4), displayed as a colored badge in the UI. A human-readable reason string explains why - e.g. "multiple sources agree; strong evidence match" or "weak retrieval match; single source only". Decided to use labels over percentages because I felt there weren't enough data points to measure confidence very precisely.

### 6. What happens when sources disagree?

The grounded prompt in `generate.py` instructs the model to present information from the provided evidence, and the `limitations` field in the JSON output is where the model notes conflicting or incomplete information. When multiple sources say different things (e.g., WHO says salt intake should be under 5g/day, NICE says under 6g/day), the model can present both figures with their respective citations.

On the confidence side, conflicting evidence naturally produces a lower source-agreement signal. Basically, if 3 sources are retrieved but they disagree, the citations won't cluster, and the confidence score reflects that. The user sees both the answer with inline references and the raw evidence chunks, so they can judge the conflict themselves.

### 7. How will uncertainty be communicated?

Three mechanisms:

- **Confidence badge**: A colored pill (green HIGH, amber MEDIUM, red LOW) with a numeric score and a reason string displayed directly above the answer. The user sees this before reading the answer.
- **Limitations field**: Every response includes a `limitations` section rendered as an info box in the UI, where the model explains what the answer doesn't cover or where evidence is thin.
- **`needs_professional` flag**: When the LLM determines the question requires professional judgment, the response gets a disclaimer appended: "This information is for educational purposes only... Always consult a qualified healthcare professional."

If no relevant evidence is found, the system doesn't attempt generation — it returns a direct message saying it couldn't find enough information and recommends consulting a professional.

### 8. How will unsafe requests be handled?

`safety.py` runs a pre-query check before any retrieval or generation happens. Two categories:

- **Emergency queries**: Regex patterns catch phrases like "having a heart attack", "can't breathe", "suicide", "overdose", "seizure right now". These immediately return a hardcoded response with emergency numbers, ensuring that no LLM involved, no latency and no risk of a bad answer.
- **Unsafe requests**: Patterns like "prescribe me", "what dose should I take", "diagnose me", "stop my medication". These return a refusal explaining that prescriptions/diagnoses require a professional, with suggestions for what the user can do instead.

Post-generation, `safety.check_response()` scans the LLM's output for prescriptive language ("you should take 500mg", "stop taking your medication") and appends a medical disclaimer if detected. This catches cases where the LLM slips past the grounded prompt constraints.

### 9. How will vague queries be processed?

Vague queries like "I feel bad" or "health tips" still go through the normal pipeline. The embedding model maps them to the nearest semantic neighborhood in the vector space, so FAISS returns whatever chunks are closest — even if the match quality is weak. The cross-encoder then rescores these, and if the reranker scores are all low (below 0), `confidence.py` caps the confidence at LOW and surfaces a reason like "weak retrieval match".

The LLM's system prompt tells it to acknowledge when evidence is insufficient rather than stretch thin results into a confident answer. The `limitations` field captures this, and `needs_professional` gets set to true. The user sees a LOW confidence badge and knows the answer isn't well-supported.

### 10. How will misspelled queries be handled?

Embedding similarity handles this naturally. When a user types "diabeties" instead of "diabetes", the sentence-transformer encodes it into a vector that's still semantically close to "diabetes" in the embedding space, since the model was trained on noisy internet text and is robust to common misspellings. FAISS cosine search then finds chunks about diabetes normally.

No explicit spell-checker or autocorrect layer is needed. The trade-off is that severely garbled queries (like "xkcd qwerty") won't match anything meaningful, but in that case the reranker scores will be low, confidence will be LOW, and the system will communicate that it couldn't find relevant information.

### 11. How will conflicting evidence be presented?

The evidence is presented transparently at multiple levels:

- **In the answer**: The LLM is instructed to reference evidence by index `[1]`, `[2]`, etc. When sources disagree, the answer can present both positions with their respective citations.
- **In citations**: Each citation card shows the source name, origin URL, and the specific claim, so the user can see which organization said what.
- **In raw evidence**: The "Retrieved Evidence Chunks" expander shows all top-5 chunks verbatim with their reranker scores and source labels. The user can read the original context and judge conflicts directly.

Confidence scoring also reflects this, because diverse sources that agree boost the score, while contradictions result in scattered citations and a lower confidence reading.

### 12. How will retrieval quality be evaluated?

- **Reranker scores**: Displayed per-chunk in the "Retrieved Evidence Chunks" expander. Cross-encoder scores above 5 indicate strong relevance, scores near 0 are marginal, negative scores mean poor matches. Users and evaluators can inspect these directly.
- **Confidence decomposition**: The confidence score breaks down into retrieval quality (normalized reranker average), source diversity, and citation coverage. The reason string in the confidence badge explains which signals were strong or weak.
- **FAISS vs rerank comparison**: The retrieval module first pulls 20 candidates by cosine similarity, then the reranker selects the best 5. Comparing the two sets shows whether reranking improved relevance. If the top FAISS result gets pushed down by the reranker, it means semantic similarity alone was misleading.
- **Source coverage**: For multi-topic queries like "lifestyle changes for hypertension", good retrieval should pull from MedQuAD, WHO, NICE, and CDC. If only one source appears, retrieval may be too narrow.
- **Edge case testing**: Queries with no knowledge base coverage should yield low reranker scores and trigger the "insufficient evidence" path. Misspelled queries should still retrieve the right chunks. These behaviors are directly observable in the UI.
