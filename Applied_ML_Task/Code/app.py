import streamlit as st
import retrieval, generate, confidence, safety

st.set_page_config(page_title="MedBridge", layout="wide")

st.markdown("""
<style>
    .main {max-width: 900px; margin: auto;}
    .stButton > button[kind="primary"] {background-color: #14b8a6; border-color: #14b8a6; color: white;}
    .stButton > button[kind="primary"]:hover {background-color: #0d9488; border-color: #0d9488;}
    .confidence-high {background: #065f46; color: white; padding: 4px 14px; border-radius: 20px; font-size: 0.85em; font-weight: 600;}
    .confidence-medium {background: #92400e; color: white; padding: 4px 14px; border-radius: 20px; font-size: 0.85em; font-weight: 600;}
    .confidence-low {background: #991b1b; color: white; padding: 4px 14px; border-radius: 20px; font-size: 0.85em; font-weight: 600;}
    .source-card {border: 1px solid #333; border-radius: 8px; padding: 12px; margin: 6px 0; background: #1a1a2e;}
    .disclaimer {color: #9ca3af; font-size: 0.8em; padding-top: 20px; border-top: 1px solid #333;}
</style>
""", unsafe_allow_html=True)

st.title("MedBridge")
st.caption("Evidence-based medical information assistant")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask a health-related question", placeholder="e.g. What lifestyle changes help with hypertension?")
submit = st.button("Search", type="primary", use_container_width=True)

if submit and query.strip():
    safety_check = safety.check_query(query)

    if not safety_check["safe"]:
        resp = safety_check["response"]
        st.warning(f"{safety_check['type'].upper()} DETECTED")
        st.markdown(resp["answer"])
        conf = {"label": "N/A", "score": 0, "reason": safety_check["type"]}
    else:
        with st.spinner("Searching medical knowledge base..."):
            normalized = generate.normalize_query(query)
            evidence = retrieval.retrieve(normalized)

        with st.spinner("Generating grounded response..."):
            result = generate.generate(normalized, evidence)

        result, warnings = safety.check_response(result)
        conf = confidence.score(evidence, result)

        for w in warnings:
            st.warning(w)

        badge_class = f"confidence-{conf['label'].lower()}"
        st.markdown(f"**Confidence:** <span class='{badge_class}'>{conf['label']} ({conf['score']})</span> — {conf['reason']}", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(result.get("answer", ""))

        if result.get("limitations"):
            st.info(f"**Limitations:** {result['limitations']}")

        citations = result.get("citations", [])
        if citations:
            st.markdown("### Citations")
            for c in citations:
                idx = c.get("source_idx", c.get("source_index", 0))
                source_name = evidence[idx - 1]["source_name"] if 0 < idx <= len(evidence) else "Unknown"
                source_url = evidence[idx - 1].get("source_url", "") if 0 < idx <= len(evidence) else ""
                st.markdown(f"""<div class='source-card'>
                    <strong>{source_name}</strong> {"· <a href='" + source_url + "'>link</a>" if source_url else ""}
                    <br><em>"{c.get('claim', '')}"</em>
                </div>""", unsafe_allow_html=True)

        with st.expander("Retrieved Evidence Chunks"):
            for i, e in enumerate(evidence):
                score_display = f"rerank: {e.get('rerank_score', 0):.3f}"
                st.markdown(f"**[{i+1}] {e['source_name']}** ({e['doc_type']}) — {score_display}")
                st.text(e["text"][:500])
                st.markdown("---")

        st.session_state.history.append({"query": query, "confidence": conf["label"]})

if st.session_state.history:
    with st.sidebar:
        st.markdown("### Query History")
        for i, h in enumerate(reversed(st.session_state.history[-10:])):
            st.markdown(f"{i+1}. {h['query'][:50]}... ({h['confidence']})")

st.markdown("<div class='disclaimer'>MedBridge is an information retrieval tool, not a medical professional. Always consult a qualified healthcare provider for personal medical decisions. Information sourced from MedQuAD, WHO, CDC, NICE, and other trusted medical organizations.</div>", unsafe_allow_html=True)
