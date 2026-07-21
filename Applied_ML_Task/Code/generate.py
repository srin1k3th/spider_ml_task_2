import json
from groq import Groq
import config

client = Groq(api_key=config.GROQ_API_KEY)

_NORMALIZE_SYSTEM = """You are a medical query normalizer. The user has typed a health question that may contain spelling mistakes, typos, or informal phrasing. Your job is to return the corrected, cleanly phrased version of the question with all spelling fixed and medical terms spelled correctly. Return ONLY the corrected question string, no explanation, no JSON, no extra text."""

def normalize_query(query):
    try:
        resp = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": _NORMALIZE_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        corrected = resp.choices[0].message.content.strip()
        return corrected if corrected else query
    except Exception:
        return query

SYSTEM_PROMPT = """You are MedBridge, a trusted medical information assistant. Your job is to answer health questions using ONLY the provided evidence chunks. You are NOT a doctor and do NOT give personal medical advice.

Writing rules:
- Write your answer as if you're explaining it to a friend — warm, clear, and easy to follow.
- Use complete sentences and short paragraphs. Break up long information into digestible sections.
- Use bullet points or numbered lists when listing symptoms, steps, or recommendations.
- If a medical term is necessary, briefly explain what it means in plain language.
- NEVER just dump raw medical text from the evidence. Always rephrase and organize it into a readable, human-friendly response.
- Focus on the evidence chunks that are MOST relevant to the user's actual question. Ignore tangentially related chunks.
- If the evidence doesn't directly address what the user asked, say so honestly rather than stretching unrelated evidence to fit.
- Reference evidence by [1], [2], etc. only when making a specific claim from that source.
- When multiple sources cover the same topic, synthesize them into one coherent answer.
- If sources disagree, present both perspectives and note the disagreement.
- Always recommend consulting a healthcare professional for personal medical decisions.

Respond in JSON with this exact structure:
{
  "answer": "Your detailed, well-written answer here. Use paragraphs and bullet points for readability. Reference evidence as [1], [2] etc.",
  "citations": [{"claim": "the specific factual claim you are citing", "source_idx": 1}],
  "limitations": "What this answer doesn't cover, where evidence is thin, or caveats the user should know.",
  "needs_professional": true/false
}"""

def build_prompt(query, evidence):
    chunks_text = ""
    for i, e in enumerate(evidence):
        chunks_text += f"\n[{i+1}] (Source: {e['source_name']})\n{e['text']}\n"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Evidence:{chunks_text}\n\nQuestion: {query}"}
    ]

def generate(query, evidence):
    if not evidence:
        return {
            "answer": "I couldn't find enough relevant medical information to answer this question reliably. Please try rephrasing your query or consult a healthcare professional.",
            "citations": [],
            "limitations": "No relevant evidence found in the knowledge base.",
            "needs_professional": True
        }
    messages = build_prompt(query, evidence)
    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=2048
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        return {
            "answer": resp.choices[0].message.content,
            "citations": [],
            "limitations": "Response parsing failed — raw text shown.",
            "needs_professional": True
        }
