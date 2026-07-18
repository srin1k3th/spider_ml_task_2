# MedBridge

A trustworthy healthcare information assistant that retrieves evidence from medical knowledge bases and generates grounded, cited responses with confidence estimation.

MedBridge is more than just a chatbot. It's a retrieval-augmented information system that finds relevant medical evidence, generates answers grounded strictly in that evidence, cites its sources, estimates reliability, and handles unsafe queries responsibly.

## Components

| Module | Purpose |
|--------|---------|
| `ingest.py` | Loads MedQuAD CSV and guideline markdown files, chunks text, saves processed JSON |
| `build_index.py` | Embeds chunks with all-MiniLM-L6-v2, builds FAISS IndexFlatIP, saves index and metadata |
| `retrieval.py` | FAISS cosine search & cross-encoder (ms-marco-MiniLM-L-6-v2) reranking |
| `generate.py` | Groq API call with grounded prompt, forces structured JSON output |
| `confidence.py` | Multi-signal scoring: retrieval quality, source agreement, citation coverage |
| `safety.py` | Pre-query emergency/unsafe detection + post-generation prescriptive language check |
| `app.py` | Streamlit UI with query input, answer panel, citation cards, confidence badge |
| `config.py` | All model names, paths, thresholds in one place |

## Design Decisions

**Why FAISS IndexFlatIP?** Corpus is <100k chunks, so exact cosine search is fast enough, thus there is no need for approximate methods. 

**Why two-stage retrieval?** Bi-encoder (all-MiniLM-L6-v2) is fast but shallow. Cross-encoder (ms-marco) is slow but accurate. FAISS narrows the chunk base, cross-encoder picks the best 5, ensuring best of both worlds.

**Why multi-signal confidence?** A single retrieval score isn't enough. If 3 different sources (WHO + CDC + NICE) all support the same claim with high reranker scores, that's much more trustworthy than a single source match. The scoring weights retrieval quality (45%), source diversity (30%), and citation coverage (25%).

**Why hardcoded safety responses?** Emergency and unsafe queries shouldn't go through the LLM at all. Pattern matching catches "I'm having a heart attack" or "prescribe me medication" before any retrieval happens, returning immediate, reliable responses with emergency numbers.

**Misspelling handling**: Embedding similarity naturally handles typos ("diabeties" is still close to "diabetes" in vector space). No explicit spell-correction is needed.

**Conflicting evidence**: The grounded prompt instructs the model to note when sources disagree and present both perspectives. Confidence drops when sources conflict.

## Evaluation Methodology

- **Retrieval quality**: Compare FAISS top-20 vs reranked top-5, ensuring whether reranking surfaces more relevant chunks?
- **Grounding**: Check that every claim in the answer maps to a retrieved evidence chunk
- **Citation correctness**: Verify source_idx values point to the right evidence
- **Safety**: Test all emergency and unsafe patterns trigger correctly
- **Hallucination check**: Ask questions with no knowledge base coverage, system should say "insufficient evidence"

## Tech Stack

- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector store**: FAISS (IndexFlatIP)
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Generation**: Groq API, Llama 3.3-70b-versatile
- **UI**: Streamlit
- **Data**: MedQuAD + WHO/CDC/NICE/USDA/HHS guidelines

## References

- MedQuad CSV
- CDC Heart Health Guidelines
- NICE Hypertension Guidelines
- USDA/HHS Nutrition Guidelines
- WHO Hypertension Guidelines
- HHS Exercise Recommendations


