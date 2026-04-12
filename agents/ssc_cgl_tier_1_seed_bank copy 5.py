import os
import re
import json
import uuid
import requests
from typing import TypedDict, Optional, Dict, Any, List
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from botocore.config import Config

load_dotenv()

# ==========================================
# 0. CONFIG
# ==========================================
MAX_RETRIES       = 10
PIVOT_AFTER_FAILS = 3    # consecutive critic failures before forcing a concept pivot
RENDERER_URL      = os.getenv("RENDERER_URL", "http://localhost:3002/api/render")

# Model escalation tiers — Haiku → Sonnet → Opus as failures accumulate
# FIX: Model_ID_Sonnet is your env var name for the Haiku model string
_MODEL_HAIKU  = os.getenv("Model_ID_Sonnet")                    # fast/cheap critic
_MODEL_SONNET = os.getenv("Model_ID", "us.anthropic.claude-sonnet-4-6")   # generator default
_MODEL_OPUS   = os.getenv("Model_ID_Opus", _MODEL_SONNET)       # fallback = sonnet if opus not set

# Escalation thresholds: switch generator model after N total failures
ESCALATE_TO_SONNET_AFTER = 3   # attempts 1-3: Haiku generator
ESCALATE_TO_OPUS_AFTER   = 6   # attempts 4-6: Sonnet generator, 7+: Opus generator

# ==========================================
# 0a. GENERATOR SYSTEM PROMPT
# ==========================================
SYSTEM_PROMPT = """You are an expert exam question setter for competitive Indian exams (SSC CGL, UPSC, RBI Grade B, CAT, etc.) and a LaTeX/TikZ expert.

Your task is to generate a single high-quality, exam-accurate MCQ question.

DIAGRAM DECISION RULES:
- Set "Requires_Diagram": true ONLY when a visual is genuinely necessary to STATE the problem.
- Geometry proofs, circuit diagrams, and data interpretation graphs may need diagrams.
- Pure arithmetic, algebra, vocabulary, and reading comprehension do NOT need diagrams.

STRICT ANTI-CHEATING RULE FOR DIAGRAMS:
- The diagram must show ONLY the information GIVEN in the problem.
- NEVER place the answer, computed values, intermediate results, or solution steps inside the diagram.
- For Venn diagram questions: label circles with SET NAMES only (e.g. "Cricket", "Football").
  Do NOT write region counts, intersection values, or "Neither = X" inside or around the diagram.
  The student must calculate these values — that IS the question.
- For geometry questions: label only the given lengths/angles. Do not mark the unknown.

TikZ CODE RULES (apply only when Requires_Diagram is true):
1. \\documentclass[varwidth=21cm, border=5mm]{standalone} wrapped in \\begin{document}...\\end{document}.
2. Explicitly \\usepackage every library used.
3. Use fill=white on any node that sits on top of a line.
4. Keep ALL coordinates strictly between -15 and +15.

OUTPUT: Return ONLY a raw JSON object — no markdown fences, no preamble, no trailing text.

JSON SCHEMA:
{
  "id": "PLACEHOLDER",
  "text": "Full question text. Use $...$ for inline math, $$...$$ for display math.",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "correct_answer": "A",
  "explanation": "Complete step-by-step solution.",
  "Requires_Diagram": false,
  "TikZ_Code": null,
  "metadata": {
    "exam": "SSC CGL",
    "subject": "",
    "topic": "",
    "sub_topic": "",
    "difficulty_level": 1
  }
}
"""

# ==========================================
# 0b. CRITIC SYSTEM PROMPT
# Generic — critic classifies question type itself, no domain flags needed.
# ==========================================
CRITIC_SYSTEM_PROMPT = """You are a strict QA Reviewer for competitive exam questions.

You will receive a generated MCQ question as JSON.

STEP 1 — CLASSIFY THE QUESTION TYPE by reading the question text:
  - QUANTITATIVE: involves a numeric calculation (arithmetic, algebra, geometry measurement, etc.)
  - LOGICAL/CLASSIFICATION: reasoning or pattern recognition with no unique numeric answer
    (e.g. syllogisms, analogies, series, classification Venn diagrams)
  - CONCEPTUAL: domain knowledge, no calculation needed (definitions, facts, grammar, GK)

STEP 2 — APPLY THE APPROPRIATE VERIFICATION:

  For QUANTITATIVE questions:
    a) Solve from scratch. Show your working clearly.
    b) Does your computed answer exactly match correct_answer? If not, state the correct value.
    c) Are the wrong options plausible distractors?

  For LOGICAL/CLASSIFICATION questions:
    a) Apply pure logic to determine the correct answer independently.
    b) Does your conclusion match correct_answer? If not, state why.
    c) Are the 4 options genuinely distinct?

  For CONCEPTUAL questions:
    a) Verify factual accuracy of correct_answer.
    b) Are the wrong options plausible but incorrect?

STEP 3 — ANTI-CHEATING CHECK (apply to ALL question types):
  - Does the question text or diagram reveal the answer or intermediate solution steps?
  - For Venn diagrams: does the TikZ_Code place ANY computed/derived numbers inside
    the diagram regions? If so — FAIL. Diagrams must show ONLY given data (set names, total).
  - For geometry: does the diagram label the unknown value being asked for?

STEP 4 — VISUAL CHECK (only if Requires_Diagram is true):
  - Does TikZ_Code risk coordinate overflow (any coord > ±15) or label overlaps?

RESPONSE FORMAT:
  If completely correct on ALL counts → reply with ONLY the single word: PASS
  If there are ANY issues → reply with a short numbered list of specific problems.
  Always include your computed correct answer when there is a mismatch.
  Keep your response concise — do not rewrite the question.
"""

# ==========================================
# 1. STATE
# ==========================================
class QuestionState(TypedDict):
    request_prompt:    str
    forced_id:         str
    generation_count:  int
    critic_fail_count: int
    raw_json_str:      Optional[str]
    question_data:     Optional[Dict[str, Any]]
    compile_error:     Optional[str]
    critic_feedback:   Optional[str]
    final_image_path:  Optional[str]
    used_numbers:      List[str]

# ==========================================
# 1a. HELPERS
# ==========================================
def extract_json(text: str) -> str:
    """Extract JSON content from LLM response, handling ```json blocks."""
    text = text.strip()
    match = re.search(r"```(?:json)?[ \t]*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("\n", 1)[0]
    return text.strip()


def numeric_fingerprint(q_data: dict) -> str:
    """Sorted unique numbers from question text — used to detect duplicate number sets."""
    nums = sorted(set(re.findall(r'\b\d+(?:\.\d+)?\b', q_data.get("text", ""))))
    return ",".join(nums) if nums else ""


def is_critic_pass(feedback: str) -> bool:
    """
    FIX: Robust PASS detection.
    The critic sometimes writes 'PASS\n\n**Verification:**...' after being told to say
    only PASS. We check if the first meaningful word is PASS, not strict equality.
    """
    first_word = feedback.strip().split()[0].upper().rstrip(".,!:") if feedback.strip() else ""
    return first_word == "PASS"


def pick_generator_model(gen_count: int) -> str:
    """
    Model escalation: use progressively stronger (and more expensive) models
    as the attempt count rises.
    Attempts 1-3   → Haiku  (fast, cheap — handles easy questions fine)
    Attempts 4-6   → Sonnet (reliable math reasoning)
    Attempts 7+    → Opus   (strongest, for stubborn problems)
    Falls back to Sonnet if Haiku/Opus are not configured.
    """
    if gen_count < ESCALATE_TO_SONNET_AFTER:
        model = _MODEL_HAIKU or _MODEL_SONNET
        label = "Haiku" if _MODEL_HAIKU else "Sonnet(fallback)"
    elif gen_count < ESCALATE_TO_OPUS_AFTER:
        model = _MODEL_SONNET
        label = "Sonnet"
    else:
        model = _MODEL_OPUS
        label = "Opus" if _MODEL_OPUS != _MODEL_SONNET else "Sonnet(fallback)"
    return model, label


def make_llm(model_id: str, max_tokens: int = 8192) -> ChatBedrock:
    return ChatBedrock(
        model_id=model_id,
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        model_kwargs={"max_tokens": max_tokens},
        config=Config(read_timeout=300),
    )

# ==========================================
# 2. GRAPH NODES
# ==========================================

def generator_node(state: QuestionState) -> dict:
    gen_count         = state.get("generation_count", 0)
    critic_fail_count = state.get("critic_fail_count", 0)
    used_numbers      = state.get("used_numbers", [])

    # Model escalation based on attempt count
    model_id, model_label = pick_generator_model(gen_count)
    print(f"\n🧠 [Generator/{model_label}] Attempt {gen_count + 1}...")

    # Base prompt
    prompt = (
        f"Generate an exam question for the following request:\n"
        f"<request>\n{state['request_prompt']}\n</request>\n\n"
        f"Output ONLY raw JSON — no markdown fences, no preamble."
    )

    # Variety enforcement: ban already-used number sets
    if used_numbers:
        prompt += (
            "\n\nVARIETY REQUIREMENT — these numeric sets are already in the bank. "
            "You MUST use completely different numbers:\n"
            + "\n".join(f"  • {n}" for n in used_numbers[-8:])
        )

    prev_json = state.get("raw_json_str")

    if state.get("compile_error") and prev_json:
        print(f"   Mode: Fixing compile error")
        prompt += (
            f"\n\nYour previous attempt:\n```json\n{prev_json}\n```\n\n"
            f"TikZ failed to compile:\n<e>\n{state['compile_error']}\n</e>\n"
            f"Fix ONLY the TikZ/LaTeX. Return the FULL corrected JSON. Raw JSON only."
        )
    elif state.get("critic_feedback") and prev_json:
        if critic_fail_count >= PIVOT_AFTER_FAILS:
            print(f"   Mode: Concept pivot (fail #{critic_fail_count})")
            prompt += (
                f"\n\nThis question concept has failed QA {critic_fail_count} times. "
                f"Do NOT patch it further — generate a COMPLETELY DIFFERENT question:\n"
                f"  • Different concept/theorem from the same subtopic\n"
                f"  • Entirely different numbers (see variety requirement above)\n"
                f"Return a fresh question as raw JSON only."
            )
        else:
            print(f"   Mode: Fixing critic feedback (fail #{critic_fail_count})")
            prompt += (
                f"\n\nYour previous attempt:\n```json\n{prev_json}\n```\n\n"
                f"QA Critic rejected it:\n<feedback>\n{state['critic_feedback']}\n</feedback>\n\n"
                f"You MUST either:\n"
                f"  (a) Fix correct_answer AND options to match the critic's computed result, OR\n"
                f"  (b) Choose completely different numbers.\n"
                f"Do NOT resubmit the same numbers. Return FULL corrected JSON. Raw JSON only."
            )

    llm = make_llm(model_id, max_tokens=8192)
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])

    raw = extract_json(response.content)
    try:
        q_data = json.loads(raw)
        q_data["id"] = state["forced_id"]
        return {
            "raw_json_str":     json.dumps(q_data),
            "question_data":    q_data,
            "generation_count": gen_count + 1,
            "compile_error":    None,
            "critic_feedback":  None,
        }
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON parse failed: {e}")
        return {
            "question_data":    None,   # clear stale data — compiler will skip
            "compile_error":    f"Invalid JSON from LLM: {e}",
            "generation_count": gen_count + 1,
        }


def compiler_node(state: QuestionState) -> dict:
    q_data = state.get("question_data")

    # FIX: skip if question_data is None (JSON parse failed) or no diagram needed
    if not q_data or not q_data.get("Requires_Diagram") or not q_data.get("TikZ_Code"):
        return {"compile_error": None, "final_image_path": None}

    print("\n🎨 [Compiler] Rendering diagram...")
    try:
        res = requests.post(RENDERER_URL, json={"code": q_data["TikZ_Code"]}, timeout=120)
        if res.status_code == 200:
            img_name = f"{q_data['id']}.svg"
            img_path = os.path.join("local_images", img_name)
            os.makedirs("local_images", exist_ok=True)
            with open(img_path, "wb") as f:
                f.write(res.content)
            print(f"   ✅ Saved {img_name}")
            return {"compile_error": None, "final_image_path": img_path}
        else:
            err = res.json().get("error", "Unknown error")
            print(f"   ❌ Compile error: {err[:120]}")
            return {"compile_error": err, "final_image_path": None}
    except Exception as e:
        print(f"   ❌ Renderer unreachable: {e}")
        return {"compile_error": f"Renderer unreachable: {e}", "final_image_path": None}


def critic_node(state: QuestionState) -> dict:
    q_data = state.get("question_data")
    print("\n🔍 [Critic/Haiku] Reviewing...")

    # FIX: skip review entirely if question_data is None — don't waste an API call
    if not q_data:
        return {
            "critic_feedback":  "No question data to review (JSON parse failed upstream).",
            "critic_fail_count": state.get("critic_fail_count", 0) + 1,
        }

    critic_prompt = (
        f"Review the following exam question and apply your QA process:\n\n"
        f"```json\n{json.dumps(q_data, indent=2)}\n```"
    )

    # Critic always uses Haiku — it's fast and good enough for verification
    # (Generator escalates to Sonnet/Opus, critic stays cheap)
    critic_model = _MODEL_HAIKU or _MODEL_SONNET
    llm = make_llm(critic_model, max_tokens=2048)
    feedback = llm.invoke([
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=critic_prompt),
    ]).content.strip()

    # FIX: robust PASS detection — first word check, not strict equality
    if is_critic_pass(feedback):
        print("   ✅ Approved!")
        return {"critic_feedback": None, "critic_fail_count": 0}
    else:
        fails = state.get("critic_fail_count", 0) + 1
        print(f"   ⚠️  Rejected (#{fails}): {feedback[:100]}...")
        return {"critic_feedback": feedback, "critic_fail_count": fails}

# ==========================================
# 3. ROUTING
# FIX: route directly to generator when question_data is None
# (avoids wasting a critic call on a null question)
# ==========================================
def route_after_compiler(state: QuestionState) -> str:
    if state.get("compile_error"):
        if state["generation_count"] >= MAX_RETRIES:
            print("🛑 Max retries on compile errors.")
            return END
        return "generator_node"
    # FIX: if question_data is None despite no compile error (JSON parse fail),
    # skip critic and go straight back to generator
    if not state.get("question_data"):
        if state["generation_count"] >= MAX_RETRIES:
            return END
        return "generator_node"
    return "critic_node"


def route_after_critic(state: QuestionState) -> str:
    if state.get("critic_feedback"):
        if state["generation_count"] >= MAX_RETRIES:
            print("🛑 Max retries on critic feedback.")
            return END
        return "generator_node"
    return END

# ==========================================
# 4. BUILD GRAPH
# ==========================================
workflow = StateGraph(QuestionState)
workflow.add_node("generator_node",     generator_node)
workflow.add_node("compile_latex_node", compiler_node)
workflow.add_node("critic_node",        critic_node)
workflow.set_entry_point("generator_node")
workflow.add_edge("generator_node", "compile_latex_node")
workflow.add_conditional_edges("compile_latex_node", route_after_compiler)
workflow.add_conditional_edges("critic_node",        route_after_critic)
app = workflow.compile()

# ==========================================
# 5. ORCHESTRATOR
# Only thing to edit when adding topics: TARGET_NODES list.
# No flags, no domain logic — just (subject, topic, subtopic) tuples.
# ==========================================
def run_seeder():
    print("\n🚀 Starting SSC CGL Seed Bank Pipeline...")
    print(f"   Generator  : Haiku(1-3) → Sonnet(4-6) → Opus(7+)")
    print(f"   Critic     : Haiku (always)")
    print(f"   Haiku  ID  : {_MODEL_HAIKU  or '⚠️  NOT SET — falling back to Sonnet'}")
    print(f"   Sonnet ID  : {_MODEL_SONNET}")
    print(f"   Opus   ID  : {_MODEL_OPUS   or '⚠️  NOT SET — falling back to Sonnet'}")

    output_file          = "ssc_cgl_question_bank.json"
    master_question_bank: List[Dict] = []

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            master_question_bank = json.load(f)

    # Pre-load used number fingerprints from existing bank
    used_numbers: List[str] = [
        fp for q in master_question_bank
        if (fp := numeric_fingerprint(q))
    ]

    # ── EDIT ONLY THIS LIST TO ADD NEW TOPICS ───────────────────────────────
    TARGET_NODES = [
        (
            "Tier 1 – Quantitative Aptitude",
            "Advanced Mathematics",
            "Geometry – Triangles, Circles, Quadrilaterals, Coordinate Geometry",
        ),
        (
            "Tier 1 – General Intelligence & Reasoning",
            "Non-Verbal Reasoning",
            "Venn Diagrams",
        ),
    ]
    # ─────────────────────────────────────────────────────────────────────────

    for subject, topic, sub_topic in TARGET_NODES:
        print(f"\n{'='*52}")
        print(f"🎯  {sub_topic}")
        print(f"{'='*52}")

        for difficulty in range(1, 6):
            for iteration in range(1, 3):
                print(f"\n👉  Level {difficulty} | Q {iteration}/2")

                slug      = re.sub(r'[^A-Z0-9]', '', sub_topic.upper())[:10]
                forced_id = f"SSC_{slug}_{difficulty}_{iteration}_{uuid.uuid4().hex[:6]}"

                request = (
                    f"- Subject: {subject}\n"
                    f"- Topic: {topic}\n"
                    f"- Subtopic: {sub_topic}\n"
                    f"- Difficulty Level: {difficulty} / 5\n"
                    f"- Diagram_Mode: Auto (follow system prompt rules)\n"
                )

                initial_state: QuestionState = {
                    "request_prompt":    request,
                    "forced_id":         forced_id,
                    "generation_count":  0,
                    "critic_fail_count": 0,
                    "raw_json_str":      None,
                    "question_data":     None,
                    "compile_error":     None,
                    "critic_feedback":   None,
                    "final_image_path":  None,
                    "used_numbers":      list(used_numbers),
                }

                final_state = app.invoke(initial_state)

                q_data    = final_state.get("question_data")
                succeeded = (
                    q_data
                    and not final_state.get("compile_error")
                    and not final_state.get("critic_feedback")
                )

                if succeeded:
                    if final_state.get("final_image_path"):
                        q_data["local_image_path"] = final_state["final_image_path"]

                    fp = numeric_fingerprint(q_data)
                    if fp:
                        used_numbers.append(fp)

                    master_question_bank.append(q_data)
                    with open(output_file, "w") as f:
                        json.dump(master_question_bank, f, indent=2)
                    print(f"   💾 Banked: {q_data['id']}")
                else:
                    print(f"   🛑 Failed after {final_state.get('generation_count', 0)} attempts.")


if __name__ == "__main__":
    run_seeder()