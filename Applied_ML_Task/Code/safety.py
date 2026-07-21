import re

EMERGENCY_PATTERNS = [
    r"\b(having a heart attack|chest pain|can'?t breathe|difficulty breathing)\b",
    r"\b(suicide|suicidal|kill myself|want to die|end my life)\b",
    r"\b(overdose|overdosed|took too many pills)\b",
    r"\b(stroke|seizure right now|choking|anaphylaxis|severe allergic)\b",
    r"\b(bleeding|uncontrolled bleeding|severe burn)\b",
]

UNSAFE_PATTERNS = [
    r"\b(prescribe me|write me a prescription|what dose)\b",
    r"\b(diagnose me|tell me my diagnosis|what disease do i have)\b",
    r"\b(should i stop taking|stop my medication|quit my meds)\b",
    r"\b(how to self.?medicate|self.?treat|treat myself without)\b",
]

PRESCRIPTIVE_PATTERNS = [
    r"\byou should take \d+\s*mg\b",
    r"\btake \d+ pills?\b",
    r"\bstop taking your (medication|medicine|prescription)\b",
    r"\byou have [A-Z][a-z]+ (disease|syndrome|disorder)\b",
]

EMERGENCY_RESPONSE = {
    "answer": "**This sounds like a medical emergency.** Please take immediate action:\n\n"
              "- **Call 112** or your local emergency number immediately\n"
              "- **KIRAN Suicide Prevention Helpline**: 1800-599-0019 (call or text)\n"
              "Do not wait. If you or someone else is in immediate danger, call emergency services right now.",
    "citations": [],
    "limitations": "This is an emergency response, not a medical consultation.",
    "needs_professional": True
}

UNSAFE_RESPONSE = {
    "answer": "I can't provide personal medical prescriptions, diagnoses, or medication changes. "
              "These decisions require a qualified healthcare professional who can evaluate your specific situation, "
              "medical history, and current medications.\n\n"
              "**What you can do:**\n"
              "- Schedule an appointment with your doctor\n"
              "- Visit an urgent care clinic if you need attention soon\n"
              "- Call a nurse helpline for guidance\n\n"
              "I'm happy to provide general health information about conditions, treatments, or lifestyle changes instead.",
    "citations": [],
    "limitations": "Cannot provide personalized medical advice.",
    "needs_professional": True
}

DISCLAIMER = ("\n\n---\n*This information is for educational purposes only and does not constitute medical advice. "
              "Always consult a qualified healthcare professional for personal medical decisions.*")

_LLM_QUERY_SYSTEM = """You are a medical safety classifier. Classify the user query into exactly one of three categories:

- emergency: the user describes an active, life-threatening situation (e.g. heart attack symptoms right now, suicidal intent, overdose, severe bleeding)
- unsafe: the user is asking for a personal prescription, diagnosis, or specific medication instruction that requires a licensed clinician
- safe: the query is a general health or medical information question that an information assistant can address

Reply with a single JSON object, no markdown:
{"category": "emergency"|"unsafe"|"safe", "reason": "one-sentence explanation"}"""

_LLM_RESPONSE_SYSTEM = """You are a medical content safety reviewer. Read the assistant response below and decide whether it contains prescriptive language — that is, specific dosage instructions, a concrete diagnosis of the user, or direct instructions to start or stop a medication.

Reply with a single JSON object, no markdown:
{"prescriptive": true|false, "reason": "one-sentence explanation"}"""

def llm_check_query(query):
    try:
        import json
        from groq import Groq
        import config
        client = Groq(api_key=config.GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": _LLM_QUERY_SYSTEM},
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=80,
        )
        data = json.loads(resp.choices[0].message.content)
        category = data.get("category", "safe").lower().strip()
        if category == "emergency":
            return {"safe": False, "type": "emergency", "response": EMERGENCY_RESPONSE,
                    "llm_reason": data.get("reason", "")}
        if category == "unsafe":
            return {"safe": False, "type": "unsafe", "response": UNSAFE_RESPONSE,
                    "llm_reason": data.get("reason", "")}
        return {"safe": True}
    except Exception:
        return None


def llm_check_response(result):
    warnings = []
    try:
        import json
        from groq import Groq
        import config
        client = Groq(api_key=config.GROQ_API_KEY)
        answer = result.get("answer", "")
        resp = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": _LLM_RESPONSE_SYSTEM},
                {"role": "user", "content": answer},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=80,
        )
        data = json.loads(resp.choices[0].message.content)
        if data.get("prescriptive", False):
            reason = data.get("reason", "prescriptive language detected by LLM")
            warnings.append(f"LLM safety check: {reason} — disclaimer added.")
            if not result["answer"].endswith(DISCLAIMER):
                result["answer"] += DISCLAIMER
    except Exception:
        pass
    return result, warnings


def check_query(query):
    q = query.lower().strip()
    for pat in EMERGENCY_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return {"safe": False, "type": "emergency", "response": EMERGENCY_RESPONSE}
    for pat in UNSAFE_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return {"safe": False, "type": "unsafe", "response": UNSAFE_RESPONSE}


    llm_result = llm_check_query(query)
    if llm_result is not None and not llm_result["safe"]:
        return llm_result

    return {"safe": True}

def check_response(result):
    answer = result.get("answer", "")
    warnings = []
    regex_triggered = False
    for pat in PRESCRIPTIVE_PATTERNS:
        if re.search(pat, answer, re.IGNORECASE):
            warnings.append("Response contained potentially prescriptive language — disclaimer added.")
            result["answer"] = answer + DISCLAIMER
            regex_triggered = True
            break


    if not regex_triggered:
        result, llm_warnings = llm_check_response(result)
        warnings.extend(llm_warnings)

    if not result["answer"].endswith(DISCLAIMER) and result.get("needs_professional", False):
        result["answer"] += DISCLAIMER
    return result, warnings
