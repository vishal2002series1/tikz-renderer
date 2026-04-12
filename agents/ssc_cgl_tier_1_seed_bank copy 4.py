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
MODEL_GENERATOR   = os.getenv("Model_ID",       "us.anthropic.claude-sonnet-4-6")

# Critic uses the faster/cheaper model when available.
# Reads Model_ID_Sonnet from .env (your Haiku model string).
# Falls back to the same Sonnet model if that var is not set.
_haiku_id   = os.getenv("Model_ID_Sonnet")   # e.g. us.anthropic.claude-haiku-4-5-20251001-v1:0
MODEL_CRITIC = _haiku_id if _haiku_id else MODEL_GENERATOR

# ==========================================
# 0a. GENERATOR SYSTEM PROMPT
# ==========================================
SYSTEM_PROMPT = """You are an expert exam question setter for competitive Indian exams (SSC CGL, UPSC, RBI Grade B, CAT, etc.) and a LaTeX/TikZ expert.

Your task is to generate a single high-quality, exam-accurate MCQ question.

DIAGRAM DECISION RULES:
- Set "Requires_Diagram": true ONLY when a visual is genuinely necessary to state or solve the problem.
- Geometry proofs, Venn diagram classification, circuit diagrams, and data interpretation graphs usually need diagrams.
- Pure arithmetic, algebra, vocabulary, and reading comprehension do NOT need diagrams.
- NEVER include a diagram that gives away the answer — if solving mentally is part of the skill being tested, keep it text-only.

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
# The critic receives the full question JSON and decides on its own
# what kind of verification is appropriate — no hardcoded flags needed.
# ==========================================
CRITIC_SYSTEM_PROMPT = """You are a strict QA Reviewer for competitive exam questions.

You will receive a generated MCQ question as JSON.

STEP 1 — CLASSIFY THE QUESTION TYPE by reading the question text and metadata:
  - QUANTITATIVE: involves a numeric calculation (arithmetic, algebra, geometry measurement, etc.)
  - LOGICAL/CLASSIFICATION: involves reasoning, classification, or pattern recognition with no unique numeric answer
    (e.g. Venn diagram "which figure best represents X, Y, Z?", syllogisms, analogies, series completion)
  - CONCEPTUAL: requires domain knowledge but no calculation (definitions, facts, grammar, GK)

STEP 2 — APPLY THE APPROPRIATE CHECK:

  For QUANTITATIVE questions:
    a) Solve the problem yourself from scratch. Show your working clearly.
    b) Does your computed answer exactly match the value in correct_answer? If not, state the correct answer.
    c) Are the wrong options plausible distractors (not trivially wrong)?

  For LOGICAL/CLASSIFICATION questions:
    a) Apply pure logic to determine the correct answer independently.
    b) Does your logical conclusion match correct_answer? If not, state why.
    c) Are the 4 options genuinely distinct from each other?

  For CONCEPTUAL questions:
    a) Verify the factual accuracy of the correct_answer.
    b) Are the wrong options plausible but clearly incorrect?

STEP 3 — CHECK ALL QUESTION TYPES:
  - CHEATING CHECK: Does the question text give away the answer (e.g. diagram shows the answer)?
  - VISUAL CHECK (only if Requires_Diagram is true): Does TikZ_Code risk coordinate overflow (coords > ±15) or label overlaps?

RESPONSE FORMAT:
  If the question is completely correct on ALL counts → reply with ONLY the single word: PASS
  If there are ANY issues → reply with a short numbered list of specific problems.
    Always include your computed/derived correct answer when there is a mismatch.
    Be concise — do not rewrite the question.
"""

# ==========================================
# 1. STATE  (no domain-specific flags)
# ==========================================
class QuestionState(TypedDict):
    request_prompt:    str
    forced_id:         str
    generation_count:  int
    critic_fail_count: int          # consecutive critic rejections for pivot logic
    raw_json_str:      Optional[str]
    question_data:     Optional[Dict[str, Any]]
    compile_error:     Optional[str]
    critic_feedback:   Optional[str]
    final_image_path:  Optional[str]
    used_numbers:      List[str]    # numeric fingerprints from banked questions

# ==========================================
# 1a. HELPERS
# ==========================================
def extract_json(text: str) -> str:
    """Extract JSON from LLM response — handles ```json blocks and raw JSON."""
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
    """
    Extract sorted unique numbers from question text as a fingerprint.
    Used to detect questions that reuse the same numeric values.
    """
    nums = sorted(set(re.findall(r'\b\d+(?:\.\d+)?\b', q_data.get("text", ""))))
    return ",".join(nums) if nums else ""


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
    print(f"\n🧠 [Generator] Attempt {gen_count + 1}...")

    # ── Base request ──
    prompt = (
        f"Generate an exam question for the following request:\n"
        f"<request>\n{state['request_prompt']}\n</request>\n\n"
        f"Output ONLY raw JSON — no markdown fences, no preamble."
    )

    # ── Variety enforcement: ban already-used number sets ──
    if used_numbers:
        prompt += (
            "\n\nVARIETY REQUIREMENT — the following numeric value sets are already "
            "in the question bank. You MUST use completely different numbers:\n"
            + "\n".join(f"  • {n}" for n in used_numbers[-8:])
        )

    prev_json = state.get("raw_json_str")

    # ── Compile error recovery ──
    if state.get("compile_error") and prev_json:
        print("🧠 [Generator] Fixing compile error...")
        prompt += (
            f"\n\nYour previous attempt produced this JSON:\n```json\n{prev_json}\n```\n\n"
            f"The TikZ code failed to compile:\n<error>\n{state['compile_error']}\n</error>\n"
            f"Fix ONLY the TikZ/LaTeX and return the FULL corrected JSON. Raw JSON only."
        )

    # ── Critic feedback recovery ──
    elif state.get("critic_feedback") and prev_json:
        if critic_fail_count >= PIVOT_AFTER_FAILS:
            # After N consecutive failures: stop patching, start fresh with a new concept
            print(f"🔀 [Generator] {critic_fail_count} consecutive rejections — pivoting concept...")
            prompt += (
                f"\n\nThis question concept has failed QA {critic_fail_count} times in a row. "
                f"Do NOT attempt to fix it further.\n"
                f"Instead, generate a COMPLETELY DIFFERENT question:\n"
                f"  • Use a different concept or theorem from the same subtopic\n"
                f"  • Use entirely different numbers (see variety requirement above)\n"
                f"  • If you were using one formula/approach, switch to another\n"
                f"Return a fresh question as raw JSON only."
            )
        else:
            print("🧠 [Generator] Fixing critic feedback...")
            prompt += (
                f"\n\nYour previous attempt produced this JSON:\n```json\n{prev_json}\n```\n\n"
                f"A QA Critic rejected it with the following feedback:\n"
                f"<feedback>\n{state['critic_feedback']}\n</feedback>\n\n"
                f"The critic has computed the correct answer above. You MUST:\n"
                f"  (a) Update correct_answer AND the option values to match exactly, OR\n"
                f"  (b) Choose completely different numbers to make the problem unambiguous.\n"
                f"Do NOT resubmit the same numbers — it will fail again.\n"
                f"Return the FULL corrected JSON. Raw JSON only."
            )

    llm = make_llm(MODEL_GENERATOR, max_tokens=8192)
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])

    raw = extract_json(response.content)
    try:
        q_data = json.loads(raw)
        q_data["id"] = state["forced_id"]  # always overwrite — never trust LLM IDs
        return {
            "raw_json_str":     json.dumps(q_data),
            "question_data":    q_data,
            "generation_count": gen_count + 1,
            "compile_error":    None,
            "critic_feedback":  None,
        }
    except json.JSONDecodeError as e:
        print(f"❌ [Generator] JSON parse failed: {e}")
        return {
            "question_data":    None,   # clear stale data so compiler skips
            "compile_error":    f"Invalid JSON from LLM: {e}",
            "generation_count": gen_count + 1,
        }


def compiler_node(state: QuestionState) -> dict:
    q_data = state.get("question_data")

    # Skip if no question, no diagram needed, or no TikZ code
    if not q_data or not q_data.get("Requires_Diagram") or not q_data.get("TikZ_Code"):
        return {"compile_error": None, "final_image_path": None}

    print("\n🎨 [Compiler] Rendering diagram via Next.js...")
    try:
        res = requests.post(RENDERER_URL, json={"code": q_data["TikZ_Code"]}, timeout=120)
        if res.status_code == 200:
            img_name = f"{q_data['id']}.svg"
            img_path = os.path.join("local_images", img_name)
            os.makedirs("local_images", exist_ok=True)
            with open(img_path, "wb") as f:
                f.write(res.content)
            print(f"✅ [Compiler] Saved {img_name}")
            return {"compile_error": None, "final_image_path": img_path}
        else:
            err = res.json().get("error", "Unknown error")
            print(f"❌ [Compiler] {err[:120]}")
            return {"compile_error": err, "final_image_path": None}
    except Exception as e:
        print(f"❌ [Compiler] Connection failed: {e}")
        return {"compile_error": f"Renderer unreachable: {e}", "final_image_path": None}


def critic_node(state: QuestionState) -> dict:
    q_data = state.get("question_data")
    print("\n🔍 [Critic] Reviewing question...")

    # The critic prompt is entirely generic — it classifies the question itself
    # and applies the right verification strategy. No subtopic flags needed here.
    critic_prompt = (
        f"Review the following exam question and apply your QA process:\n\n"
        f"```json\n{json.dumps(q_data, indent=2)}\n```"
    )

    llm = make_llm(MODEL_CRITIC, max_tokens=2048)
    feedback = llm.invoke([
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=critic_prompt),
    ]).content.strip()

    if feedback.upper().strip() == "PASS":
        print("✅ [Critic] Approved!")
        return {"critic_feedback": None, "critic_fail_count": 0}
    else:
        fails = state.get("critic_fail_count", 0) + 1
        print(f"⚠️ [Critic] Rejected (#{fails}): {feedback[:100]}...")
        return {"critic_feedback": feedback, "critic_fail_count": fails}

# ==========================================
# 3. ROUTING
# ==========================================
def route_after_compiler(state: QuestionState) -> str:
    if state.get("compile_error"):
        if state["generation_count"] >= MAX_RETRIES:
            print("🛑 Max retries on compile errors.")
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
# The only thing you ever change is TARGET_NODES below.
# No flags, no domain logic — just (subject, topic, subtopic) tuples.
# ==========================================
def run_seeder():
    print("\n🚀 Starting SSC CGL Seed Bank Pipeline...")
    print(f"   Generator : {MODEL_GENERATOR}")
    print(f"   Critic    : {MODEL_CRITIC}")
    if MODEL_CRITIC == MODEL_GENERATOR:
        print("   ⚠️  Model_ID_Sonnet not set in .env — critic falling back to Sonnet (slower/costlier)")

    output_file          = "ssc_cgl_question_bank.json"
    master_question_bank: List[Dict] = []

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            master_question_bank = json.load(f)

    # Pre-load used number fingerprints from existing bank
    # so resumed runs don't repeat numbers already in the file
    used_numbers: List[str] = [
        fp for q in master_question_bank
        if (fp := numeric_fingerprint(q))
    ]

    # ── ONLY THING TO EDIT WHEN ADDING NEW TOPICS ──────────────────────────
    # Plain (subject, topic, subtopic) tuples — no flags, no domain knowledge
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
    # ───────────────────────────────────────────────────────────────────────

    for subject, topic, sub_topic in TARGET_NODES:
        print(f"\n{'='*50}")
        print(f"🎯  {sub_topic}")
        print(f"{'='*50}")

        for difficulty in range(1, 6):
            for iteration in range(1, 3):
                print(f"\n👉  {sub_topic} | Level {difficulty} | Q {iteration}/2")

                # Deterministic unique ID — never from the LLM
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

                    # Record fingerprint so next question avoids these numbers
                    fp = numeric_fingerprint(q_data)
                    if fp:
                        used_numbers.append(fp)

                    master_question_bank.append(q_data)
                    with open(output_file, "w") as f:
                        json.dump(master_question_bank, f, indent=2)
                    print(f"💾  Banked: {q_data['id']}")
                else:
                    print(f"🛑  Failed after {final_state.get('generation_count', 0)} attempts.")


if __name__ == "__main__":
    run_seeder()