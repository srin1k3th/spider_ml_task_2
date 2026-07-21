import config

def score(evidence, result):
    if not evidence:
        return {"label": "LOW", "score": 0.0, "reason": "No evidence retrieved"}

    rerank_scores = [e.get("rerank_score", 0) for e in evidence]
    avg_rerank = sum(rerank_scores) / len(rerank_scores)
    max_rerank = max(rerank_scores)

    sources = set(e["source_name"] for e in evidence)
    source_count = len(sources)

    citation_count = len(result.get("citations", []))

    # normalize rerank score — widen the range so conversational queries aren't penalized as harshly
    # ms-marco cross-encoder scores typically range from -10 to 10
    norm_rerank = max(0, min(1, (avg_rerank + 10) / 20))

    source_score = min(1.0, source_count / 3)
    citation_score = min(1.0, citation_count / 3)

    final = 0.40 * norm_rerank + 0.30 * source_score + 0.30 * citation_score

    # gentle penalty instead of hard cap when reranker scores are all negative
    if max_rerank < 0:
        final = final * 0.7

    if final >= config.CONFIDENCE_THRESHOLDS["high"]:
        label = "HIGH"
    elif final >= config.CONFIDENCE_THRESHOLDS["medium"]:
        label = "MEDIUM"
    else:
        label = "LOW"

    reasons = []
    if norm_rerank < 0.3:
        reasons.append("weak retrieval match")
    if source_count == 1:
        reasons.append("single source only")
    if citation_count == 0:
        reasons.append("no specific citations generated")
    if max_rerank > 5:
        reasons.append("strong evidence match")
    if source_count >= 3:
        reasons.append("multiple sources agree")
    if citation_count >= 3:
        reasons.append("well-cited response")

    return {
        "label": label,
        "score": round(final, 3),
        "reason": "; ".join(reasons) if reasons else "adequate evidence"
    }
