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


