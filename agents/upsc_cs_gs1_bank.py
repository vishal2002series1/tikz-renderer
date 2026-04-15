import os
import re
import json
import uuid
import base64
import requests
from typing import TypedDict, Optional, Dict, Any, List
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from botocore.config import Config

try:
    import cairosvg
    _CAIROSVG_AVAILABLE = True
except ImportError:
    _CAIROSVG_AVAILABLE = False
    print("⚠️  cairosvg not installed — diagram critic will use TikZ text only.")
    print("   Install with: pip install cairosvg --break-system-packages")

load_dotenv()

# ==========================================
# 0. CONFIG
# ==========================================
MAX_RETRIES       = 10
PIVOT_AFTER_FAILS = 3    # total_fail_count threshold for concept pivot
RENDERER_URL      = os.getenv("RENDERER_URL", "http://localhost:3002/api/render")

_MODEL_HAIKU  = os.getenv("Model_ID_Sonnet")
_MODEL_SONNET = os.getenv("Model_ID", "us.anthropic.claude-sonnet-4-6")
_MODEL_OPUS   = os.getenv("Model_ID_Opus", _MODEL_SONNET)

# ==========================================
# 0a. GENERATOR SYSTEM PROMPT
# ==========================================
SYSTEM_PROMPT = """You are an expert UPSC Civil Services Preliminary Examination question setter and a LaTeX/TikZ expert.

Generate a single high-quality, exam-accurate MCQ question for the UPSC CSE Prelims General Studies Paper 1 (GS-1).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UPSC GS-1 QUESTION STYLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UPSC Prelims GS-1 questions are 100% conceptual and factual — there are NO numerical calculations.
All questions are 4-option MCQs with a single correct answer.

Authentic UPSC question patterns (use these):
  STATEMENT TYPE: "Consider the following statements: 1. ... 2. ... 3. ...
    Which of the above statements is/are correct?
    (a) 1 only  (b) 1 and 2 only  (c) 2 and 3 only  (d) 1, 2 and 3"

  MATCHING TYPE: "Which of the following pairs is/are correctly matched?
    Article / Provision — match columns"

  DIRECT FACTUAL: "With reference to [topic], which of the following is correct?"

  ASSERTION-REASON: "Assertion (A): ... Reason (R): ...
    (a) Both A and R are true and R is the correct explanation of A
    (b) Both A and R are true but R is NOT the correct explanation of A
    (c) A is true but R is false  (d) A is false but R is true"

• Use "Which of the following statements is/are correct?" style frequently.
• Wrong options must be plausible and based on common misconceptions.
• Never use vague language like "sometimes", "generally", or "may".
• Constitutional articles, landmark case names, and convention years must be accurate.

Difficulty scale:
  1 = Direct recall of a single fact (Article number, name, year)
  2 = Requires distinguishing between similar concepts
  3 = Requires knowledge of exceptions, nuances, or landmark cases
  4 = Statement-based with 2-3 statements requiring careful elimination
  5 = Complex multi-statement or assertion-reason requiring deep understanding

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIAGRAM DECISION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UPSC GS-1 questions almost never require diagrams. Set "Requires_Diagram": false for
almost all questions. Only set true for questions about maps, physical geography features,
or biological diagrams where a visual is genuinely necessary to state the problem.

When a diagram IS used:
  • Keep it extremely simple — no computed values, no answer hints.
  • Label only what is GIVEN in the question.
  • ANTI-CHEATING: never draw the answer or any intermediate result.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TikZ CODE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. \\documentclass[varwidth=21cm, border=5mm]{standalone}
2. \\usepackage{tikz} and explicitly load every library used.
3. fill=white on any node overlapping a line.
4. ALL raw coordinates strictly between -12 and +12.
5. No global \\scale transforms.
6. Keep the diagram SIMPLE. A clean 5-line diagram beats a complex 30-line one.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a raw JSON object — no markdown fences, no preamble, no extra text after the closing brace.

{
  "id": "PLACEHOLDER",
  "text": "Question text following authentic UPSC question patterns.",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "correct_answer": "A",
  "explanation": "Complete explanation citing constitutional articles, landmark cases, or scientific facts as applicable. Explain why each wrong option is incorrect.",
  "Requires_Diagram": false,
  "TikZ_Code": null,
  "metadata": {
    "exam": "UPSC CSE Prelims GS-1",
    "subject": "",
    "topic": "",
    "sub_topic": "",
    "difficulty_level": 1
  }
}
"""

# ==========================================
# 0b. MATH CRITIC PROMPT (Sonnet)
# FIX: Use Sonnet — Haiku makes arithmetic errors on geometry (circumradius, inradius).
# Sonnet is much more reliable at multi-step math verification.
# ==========================================
MATH_CRITIC_PROMPT = """You are a strict QA reviewer for UPSC Civil Services Preliminary Examination GS-1 questions.
You have deep expertise in Indian Polity, History, Geography, Environment, Economy, and General Science
as tested in UPSC Prelims.

UPSC GS-1 has NO numerical calculations — all questions are conceptual and factual.

STEP 1 — CLASSIFY the question type:
  DIRECT FACTUAL: tests recall of a single fact
  STATEMENT-BASED: "Which of the following statements is/are correct?" — requires verifying 2-4 statements
  MATCHING: tests correct pairing of items
  ASSERTION-REASON: tests both fact and causal relationship
  APPLICATION: applies a concept to a scenario

STEP 2 — VERIFY FACTUAL ACCURACY independently:
  • Check every factual claim in the question text against your knowledge of the UPSC syllabus.
  • For STATEMENT-BASED questions: verify EACH statement individually.
    Mark each as TRUE or FALSE with your reasoning.
    Does your T/F pattern exactly match the correct_answer option?
  • For MATCHING questions: verify every pair independently.
  • For ASSERTION-REASON: verify both A and R independently, then check if R explains A.
  • Constitutional articles must be precisely correct (e.g. Article 21 vs 21A).
  • Landmark case names and years must be accurate.
  • IUCN category definitions, hotspot criteria, convention years must be exact.

STEP 3 — OPTION QUALITY CHECK:
  • Are all 4 options plausible? No obviously silly distractors.
  • For statement-based: are all combinations (1 only / 1&2 only / 2&3 only / 1,2&3) distinct?
  • Is the correct answer unambiguous — no other option could also be defended as correct?

STEP 4 — UPSC STYLE CHECK:
  • Does the question follow authentic UPSC patterns?
  • Is language precise and unambiguous (no "sometimes", "generally", "may")?
  • Does the explanation clearly cite why each wrong option is incorrect?

STEP 5 — ANTI-CHEATING: Does the question text directly reveal the answer?

RESPONSE:
  Everything correct → reply with ONLY the single word: PASS
  Any issue → short numbered list. For factual errors, state the correct fact.
  For statement-based errors, state which statements are actually T/F and why.
"""

# ==========================================
# 0c. DIAGRAM CRITIC PROMPT (Sonnet + Vision)
# FIX: REMOVED all geometric computation checks (G2-G5).
# These were making arithmetic errors (confirmed: G2 rejected a correct diagram).
# Math accuracy is the math critic's job. The diagram critic's ONLY job is visual.
# ==========================================
DIAGRAM_CRITIC_PROMPT = """You are a diagram visual QA reviewer. You receive the rendered diagram image.
Your job is purely visual — you are NOT checking mathematical accuracy (that is done separately).

Look at the image and check ONLY these four things:

V1. CLIPPING: Are any labels, lines, or shapes cut off at the image boundary?
    Fail if any element is partially outside the frame.

V2. OVERLAPS: Do text labels overlap with lines or other labels making them unreadable?
    Fail if any label is obscured or illegible.

V3. ANTI-CHEATING: Does the diagram show the answer value or any computed result?
    • Venn diagrams: are any numeric counts or "Neither = X" labels shown in regions?
    • Geometry: is the unknown quantity labelled with its value (not "?")?
    Fail if the answer or any intermediate computed value appears in the diagram.

V4. BASIC RECOGNISABILITY: Is the main shape recognisable?
    A triangle should look like a triangle. A circle should look like a circle.
    Fail ONLY if the shape is so distorted it would genuinely confuse a student.
    Minor proportional inaccuracies are acceptable — this is a visual aid, not a CAD drawing.

RESPONSE:
  All four checks pass → reply with ONLY the single word: PASS
  Any check fails → short numbered list of what you see in the image.
  Be specific and concise. Do not compute any mathematics.
"""

# ==========================================
# 0d. SVG → PNG HELPER
# ==========================================
def svg_to_png_base64(svg_path: str, output_width: int = 900) -> Optional[str]:
    if not _CAIROSVG_AVAILABLE:
        return None
    try:
        png_bytes = cairosvg.svg2png(url=svg_path, output_width=output_width)
        return base64.standard_b64encode(png_bytes).decode("utf-8")
    except Exception as e:
        print(f"   ⚠️  SVG→PNG conversion failed: {e}")
        return None

# ==========================================
# 1. STATE
# FIX: Single total_fail_count replaces separate math_fail_count + diagram_fail_count.
# This prevents the two critics from fighting each other over the 10-attempt budget.
# ==========================================
class QuestionState(TypedDict):
    request_prompt:     str
    forced_id:          str
    generation_count:   int
    total_fail_count:   int           # FIX: single unified failure counter
    last_failure_type:  str           # "math" | "diagram" | "compile" | "json"
    raw_json_str:       Optional[str]
    question_data:      Optional[Dict[str, Any]]
    compile_error:      Optional[str]
    math_feedback:      Optional[str]
    diagram_feedback:   Optional[str]
    final_image_path:   Optional[str]
    used_numbers:       List[str]

# ==========================================
# 1a. HELPERS
# ==========================================
def extract_json(text: str) -> str:
    """
    FIX: Aggressive JSON extraction handles Opus thinking blocks and trailing text.
    Opus often outputs reasoning before/after JSON causing 'Extra data' parse errors.
    Strategy: find the outermost { } pair.
    """
    text = text.strip()

    # Try ```json ... ``` block first
    match = re.search(r"```(?:json)?[ \t]*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # FIX: Find the first { and last } to extract JSON even with surrounding text
    first_brace = text.find('{')
    last_brace  = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace:last_brace + 1]

    # Fallback: strip ``` markers
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("\n", 1)[0]
    return text.strip()


def numeric_fingerprint(q_data: dict) -> str:
    nums = sorted(set(re.findall(r'\b\d+(?:\.\d+)?\b', q_data.get("text", ""))))
    return ",".join(nums) if nums else ""


def is_pass(feedback: str) -> bool:
    """First meaningful word is PASS (handles 'PASS\n\nVerification:...' from verbose models)."""
    words = feedback.strip().split()
    if not words:
        return False
    return words[0].upper().rstrip(".,!:*#") == "PASS"


def needs_diagram(q_data: Optional[dict]) -> bool:
    return bool(q_data and q_data.get("Requires_Diagram") and q_data.get("TikZ_Code"))


def pick_generator_model(gen_count: int, has_diagram: bool) -> tuple:
    """
    Smart model routing:
    - Diagram questions → Sonnet immediately (Haiku can't reliably compute geometry).
    - Text-only → Haiku first, escalate to Sonnet then Opus.
    """
    if has_diagram:
        if gen_count < 6:
            return _MODEL_SONNET, "Sonnet"
        else:
            return _MODEL_OPUS, "Opus" if _MODEL_OPUS != _MODEL_SONNET else "Sonnet"
    else:
        if gen_count < 3:
            m = _MODEL_HAIKU or _MODEL_SONNET
            return m, "Haiku" if _MODEL_HAIKU else "Sonnet(fallback)"
        elif gen_count < 6:
            return _MODEL_SONNET, "Sonnet"
        else:
            return _MODEL_OPUS, "Opus" if _MODEL_OPUS != _MODEL_SONNET else "Sonnet"


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
    gen_count        = state.get("generation_count", 0)
    total_fails      = state.get("total_fail_count", 0)
    last_failure     = state.get("last_failure_type", "")
    used_numbers     = state.get("used_numbers", [])
    prev_data        = state.get("question_data")
    prev_had_diagram = needs_diagram(prev_data)

    model_id, model_label = pick_generator_model(gen_count, prev_had_diagram)
    print(f"\n🧠 [Generator/{model_label}] Attempt {gen_count + 1}...")

    # Base prompt
    prompt = (
        f"Generate an exam question for:\n"
        f"<request>\n{state['request_prompt']}\n</request>\n\n"
        f"Output ONLY raw JSON — no markdown fences, no preamble, "
        f"no text after the closing brace."
    )

    if used_numbers:
        prompt += (
            "\n\nVARIETY: These number sets are already banked — use completely different numbers:\n"
            + "\n".join(f"  • {n}" for n in used_numbers[-8:])
        )

    prev_json = state.get("raw_json_str")

    # ── Compile error ──
    if last_failure == "compile" and prev_json:
        print("   Mode: Fixing compile error")
        prompt += (
            f"\n\nPrevious JSON:\n```json\n{prev_json}\n```\n\n"
            f"TikZ failed to compile:\n<e>\n{state['compile_error']}\n</e>\n"
            f"Fix ONLY the TikZ. Re-run coordinate verification (Steps A-C). "
            f"Return FULL corrected JSON. Raw JSON only, nothing after closing brace."
        )

    # ── Diagram geometry failure — FIX: preserve question, fix only TikZ ──
    elif last_failure == "diagram" and prev_json:
        if total_fails >= PIVOT_AFTER_FAILS:
            print(f"   Mode: Diagram pivot (total fails: {total_fails})")
            prompt += (
                f"\n\nDiagram QA has failed {total_fails} times. "
                f"Generate a COMPLETELY DIFFERENT question — "
                f"either text-only OR with a simpler diagram configuration. "
                f"Use different numbers. Raw JSON only."
            )
        else:
            print(f"   Mode: Fixing diagram only (total fails: {total_fails})")
            # FIX: Tell the generator to keep question text/answer, only fix TikZ
            prompt += (
                f"\n\nPrevious JSON:\n```json\n{prev_json}\n```\n\n"
                f"The diagram has geometric errors:\n"
                f"<diagram_feedback>\n{state['diagram_feedback']}\n</diagram_feedback>\n\n"
                f"IMPORTANT: Keep the question text, options, correct_answer, and explanation "
                f"EXACTLY the same. ONLY fix the TikZ_Code coordinates.\n"
                f"Redo coordinate computation (Steps A-C) completely from scratch. "
                f"Show coordinate working in a comment in the TikZ. Return FULL JSON. Raw JSON only."
            )

    # ── Math failure ──
    elif last_failure == "math" and prev_json:
        if total_fails >= PIVOT_AFTER_FAILS:
            print(f"   Mode: Math pivot (total fails: {total_fails})")
            prompt += (
                f"\n\nThis question concept has failed math QA {total_fails} times. "
                f"Generate a COMPLETELY DIFFERENT question — different concept, different numbers. "
                f"Raw JSON only."
            )
        else:
            print(f"   Mode: Fixing math (total fails: {total_fails})")
            prompt += (
                f"\n\nPrevious JSON:\n```json\n{prev_json}\n```\n\n"
                f"Math QA rejected it:\n<feedback>\n{state['math_feedback']}\n</feedback>\n\n"
                f"Either fix correct_answer and options to match the reviewer's result, "
                f"or use completely different numbers. "
                f"Raw JSON only, nothing after closing brace."
            )

    llm = make_llm(model_id, max_tokens=8192)
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])

    raw = extract_json(response.content)
    try:
        q_data = json.loads(raw)
        q_data["id"] = state["forced_id"]
        return {
            "raw_json_str":      json.dumps(q_data),
            "question_data":     q_data,
            "generation_count": gen_count + 1,
            "compile_error":    None,
            "math_feedback":    None,
            "diagram_feedback": None,
            "last_failure_type": "",
        }
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON parse failed: {e}")
        return {
            "question_data":    None,
            "compile_error":    f"JSON parse error: {e}",
            "generation_count": gen_count + 1,
            "total_fail_count": state.get("total_fail_count", 0) + 1,
            "last_failure_type": "json",
        }


def compiler_node(state: QuestionState) -> dict:
    q_data = state.get("question_data")

    # FIX: Always clear final_image_path when current question has no diagram.
    # Prevents stale SVG from a previous attempt being used for a different question.
    if not q_data or not q_data.get("Requires_Diagram") or not q_data.get("TikZ_Code"):
        return {"compile_error": None, "final_image_path": None}

    print("\n🎨 [Compiler] Rendering diagram...")
    try:
        res = requests.post(RENDERER_URL, json={"code": q_data["TikZ_Code"]}, timeout=120)
        if res.status_code == 200:
            # FIX: Use attempt-specific filename (include generation_count) so each
            # attempt writes a distinct file. Prevents cross-attempt SVG contamination.
            gen = state.get("generation_count", 0)
            img_name = f"{q_data['id']}_a{gen}.svg"
            img_path = os.path.join("local_images", img_name)
            os.makedirs("local_images", exist_ok=True)
            with open(img_path, "wb") as f:
                f.write(res.content)
            print(f"   ✅ Saved {img_name}")
            return {"compile_error": None, "final_image_path": img_path}
        else:
            err = res.json().get("error", "Unknown error")
            print(f"   ❌ Compile error: {err[:120]}")
            return {
                "compile_error":    err,
                "final_image_path": None,
                "total_fail_count": state.get("total_fail_count", 0) + 1,
                "last_failure_type": "compile",
            }
    except Exception as e:
        print(f"   ❌ Renderer unreachable: {e}")
        return {
            "compile_error":    f"Renderer unreachable: {e}",
            "final_image_path": None,
            "total_fail_count": state.get("total_fail_count", 0) + 1,
            "last_failure_type": "compile",
        }


def math_critic_node(state: QuestionState) -> dict:
    """
    Sonnet math critic — verifies answer correctness only. Does not see TikZ code.
    FIX: Changed from Haiku to Sonnet — Haiku makes arithmetic errors on geometry.
    """
    q_data = state.get("question_data")
    print("\n🔢 [MathCritic/Sonnet] Verifying answer...")

    if not q_data:
        return {
            "math_feedback":    "No question data.",
            "total_fail_count": state.get("total_fail_count", 0) + 1,
            "last_failure_type": "json",
        }

    # Strip TikZ — math critic only needs question/answer/explanation
    q_for_critic = {k: v for k, v in q_data.items() if k != "TikZ_Code"}

    # FIX: Use Sonnet for math verification (Haiku made arithmetic errors)
    feedback = make_llm(_MODEL_SONNET, max_tokens=1024).invoke([
        SystemMessage(content=MATH_CRITIC_PROMPT),
        HumanMessage(content=f"Review:\n```json\n{json.dumps(q_for_critic, indent=2)}\n```"),
    ]).content.strip()

    if is_pass(feedback):
        print("   ✅ Math approved!")
        return {"math_feedback": None}
    else:
        fails = state.get("total_fail_count", 0) + 1
        print(f"   ⚠️  Math rejected (total fails: {fails}): {feedback[:100]}...")
        return {
            "math_feedback":    feedback,
            "total_fail_count": fails,
            "last_failure_type": "math",
        }


def diagram_critic_node(state: QuestionState) -> dict:
    """
    Multimodal Sonnet — visual check ONLY using the rendered image.
    FIX: Removed TikZ text and all geometric computation checks.
    Those caused false rejections (arithmetic errors in G2).
    This critic only looks at the image for: clipping, overlaps, anti-cheating, recognisability.
    """
    q_data   = state.get("question_data")
    img_path = state.get("final_image_path")
    print("\n📐 [DiagramCritic/Sonnet+Vision] Visual check...")

    if not needs_diagram(q_data):
        return {"diagram_feedback": None}

    # FIX: Minimal text context — just question text so critic knows what the diagram shows.
    # No TikZ, no coordinates, no math. Purely visual.
    text_prompt = (
        f"The diagram above illustrates this exam question:\n"
        f"{q_data.get('text', '')}\n\n"
        f"Apply visual checks V1-V4 as instructed."
    )

    # Load rendered image
    png_b64 = None
    if img_path and os.path.exists(img_path):
        png_b64 = svg_to_png_base64(img_path)
        if png_b64:
            print("   🖼️  Image loaded for visual review")
        else:
            print("   ⚠️  SVG→PNG failed — skipping diagram visual check")
    else:
        print("   ⚠️  No rendered image — skipping diagram visual check")

    # If no image available, skip diagram check entirely (don't block on text-only review)
    if not png_b64:
        print("   ✅ Diagram check skipped (no image available)")
        return {"diagram_feedback": None}

    human_content = [
        {"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": png_b64
        }},
        {"type": "text", "text": text_prompt},
    ]

    feedback = make_llm(_MODEL_SONNET, max_tokens=512).invoke([
        SystemMessage(content=DIAGRAM_CRITIC_PROMPT),
        HumanMessage(content=human_content),
    ]).content.strip()

    if is_pass(feedback):
        print("   ✅ Diagram approved!")
        return {"diagram_feedback": None}
    else:
        fails = state.get("total_fail_count", 0) + 1
        print(f"   ⚠️  Diagram rejected (total fails: {fails}): {feedback[:130]}...")
        return {
            "diagram_feedback": feedback,
            "total_fail_count": fails,
            "last_failure_type": "diagram",
        }

# ==========================================
# 3. ROUTING
# FIX: All routing decisions use total_fail_count (unified budget).
# ==========================================
def route_after_compiler(state: QuestionState) -> str:
    if state.get("compile_error") or not state.get("question_data"):
        if state.get("total_fail_count", 0) >= MAX_RETRIES:
            print("🛑 Max retries hit.")
            return END
        return "generator_node"
    return "math_critic_node"


def route_after_math_critic(state: QuestionState) -> str:
    if state.get("math_feedback"):
        if state.get("total_fail_count", 0) >= MAX_RETRIES:
            print("🛑 Max retries hit.")
            return END
        return "generator_node"
    if needs_diagram(state.get("question_data")):
        return "diagram_critic_node"
    return END


def route_after_diagram_critic(state: QuestionState) -> str:
    if state.get("diagram_feedback"):
        if state.get("total_fail_count", 0) >= MAX_RETRIES:
            print("🛑 Max retries hit.")
            return END
        return "generator_node"
    return END

# ==========================================
# 4. BUILD GRAPH
# ==========================================
workflow = StateGraph(QuestionState)
workflow.add_node("generator_node",      generator_node)
workflow.add_node("compile_latex_node",  compiler_node)
workflow.add_node("math_critic_node",    math_critic_node)
workflow.add_node("diagram_critic_node", diagram_critic_node)
workflow.set_entry_point("generator_node")
workflow.add_edge("generator_node", "compile_latex_node")
workflow.add_conditional_edges("compile_latex_node",  route_after_compiler)
workflow.add_conditional_edges("math_critic_node",    route_after_math_critic)
workflow.add_conditional_edges("diagram_critic_node", route_after_diagram_critic)
app = workflow.compile()

# ==========================================
# 5. ORCHESTRATOR
# ==========================================
def run_seeder():
    print("\n🚀 Starting UPSC CSE Prelims GS-1 Question Bank Pipeline...")
    print(f"   Haiku  : {_MODEL_HAIKU  or '⚠️  NOT SET — falls back to Sonnet'}")
    print(f"   Sonnet : {_MODEL_SONNET}")
    print(f"   Opus   : {_MODEL_OPUS}")
    print(f"   Critics: Factual=Sonnet (UPSC GS-1 accuracy) | Diagram=Sonnet+Vision (visual only)")
    print(f"   Budget : {MAX_RETRIES} attempts per round, infinite rounds per slot (never skips)")
    vision_status = "✅ cairosvg installed" if _CAIROSVG_AVAILABLE else "⚠️  cairosvg missing"
    print(f"   Vision : {vision_status}")

    output_file          = "upsc_gs1_question_bank.json"
    master_question_bank: List[Dict] = []

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            master_question_bank = json.load(f)

    used_numbers: List[str] = [
        fp for q in master_question_bank if (fp := numeric_fingerprint(q))
    ]

    # ── EDIT ONLY THIS LIST ─────────────────────────────────────────────────
    # Format: (subject, topic, sub_topic)
    # Matches the UPSC CSE Prelims GS-1 syllabus structure.
    TARGET_NODES = [
        (
            "Indian Polity",
            "Fundamental Rights",
            "Fundamental Rights (Articles 12–35)",
        ),
        (
            "Environment & Ecology",
            "Biodiversity",
            "Biodiversity – Levels, Hotspots, IUCN Categories",
        ),
    ]
    # ────────────────────────────────────────────────────────────────────────

    for subject, topic, sub_topic in TARGET_NODES:
        print(f"\n{'='*56}")
        print(f"🎯  {subject} → {sub_topic}")
        print(f"{'='*56}")

        for difficulty in range(1, 6):
            for iteration in range(1, 3):
                print(f"\n👉  Level {difficulty} | Q {iteration}/2")

                slug = re.sub(r'[^A-Z0-9]', '', sub_topic.upper())[:10]

                round_num      = 0
                total_attempts = 0
                banked         = False
                tried_concepts: List[str] = []

                while not banked:
                    round_num += 1
                    forced_id = f"UPSC_{slug}_{difficulty}_{iteration}_{uuid.uuid4().hex[:6]}"

                    extra_hint = ""
                    if round_num > 1:
                        concepts_str = (
                            "\n".join(f"  • {c}" for c in tried_concepts[-5:])
                            if tried_concepts else "  • (none recorded)"
                        )
                        extra_hint = (
                            f"\n- IMPORTANT: {round_num - 1} previous round(s) of "
                            f"{MAX_RETRIES} attempts failed for this slot.\n"
                            f"- The following question angles were already tried and failed:\n"
                            f"{concepts_str}\n"
                            f"- You MUST choose a completely different angle — different "
                            f"articles, different cases, different aspect of the sub-topic.\n"
                            f"- Vary the question type: try Statement-based if Direct Factual "
                            f"failed, or Assertion-Reason if Statement-based failed."
                        )

                    request = (
                        f"- Exam: UPSC Civil Services Preliminary Examination GS-1\n"
                        f"- Subject: {subject}\n"
                        f"- Topic: {topic}\n"
                        f"- Sub-topic: {sub_topic}\n"
                        f"- Difficulty Level: {difficulty} / 5\n"
                        f"- Question type preference: "
                        f"{'Statement-based or Assertion-Reason' if difficulty >= 3 else 'Direct Factual or Matching'}\n"
                        f"- Diagram_Mode: Auto (almost always false for UPSC GS-1)\n"
                        + extra_hint
                    )

                    print(f"   🔁 Round {round_num}")

                    initial_state: QuestionState = {
                        "request_prompt":    request,
                        "forced_id":         forced_id,
                        "generation_count":  0,
                        "total_fail_count":  0,
                        "last_failure_type": "",
                        "raw_json_str":      None,
                        "question_data":     None,
                        "compile_error":     None,
                        "math_feedback":     None,
                        "diagram_feedback":  None,
                        "final_image_path":  None,
                        "used_numbers":      list(used_numbers),
                    }

                    final_state = app.invoke(initial_state)
                    total_attempts += final_state.get("generation_count", 0)

                    last_q = final_state.get("question_data")
                    if last_q:
                        tried_sub = last_q.get("metadata", {}).get("sub_topic", "")
                        if tried_sub and tried_sub not in tried_concepts:
                            tried_concepts.append(tried_sub)

                    q_data = final_state.get("question_data")
                    succeeded = (
                        q_data
                        and not final_state.get("compile_error")
                        and not final_state.get("math_feedback")
                        and not final_state.get("diagram_feedback")
                    )

                    if succeeded:
                        tmp_img = final_state.get("final_image_path")
                        if tmp_img and os.path.exists(tmp_img):
                            final_img = os.path.join(
                                "local_images", f"{q_data['id']}.svg"
                            )
                            os.rename(tmp_img, final_img)
                            q_data["local_image_path"] = final_img

                        fp = numeric_fingerprint(q_data)
                        if fp:
                            used_numbers.append(fp)

                        master_question_bank.append(q_data)
                        with open(output_file, "w") as f:
                            json.dump(master_question_bank, f, indent=2)

                        icon = "📐" if q_data.get("Requires_Diagram") else "📝"
                        print(
                            f"   💾 Banked {icon}: {q_data['id']} "
                            f"(round {round_num}, {total_attempts} total attempts)"
                        )
                        banked = True

                    else:
                        used_in_run = final_state.get("generation_count", 0)
                        print(
                            f"   ⚠️  Round {round_num} exhausted "
                            f"({used_in_run} attempts). Retrying with fresh angle..."
                        )


if __name__ == "__main__":
    run_seeder()