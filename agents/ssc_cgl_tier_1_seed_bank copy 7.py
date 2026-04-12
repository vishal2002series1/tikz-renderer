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
SYSTEM_PROMPT = """You are an expert exam question setter for competitive Indian exams and a LaTeX/TikZ expert.

Generate a single high-quality, exam-accurate MCQ question in the specified subject/topic.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIAGRAM DECISION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Only set "Requires_Diagram": true when a visual genuinely helps state the problem.
The diagram is a VISUAL AID, not a precision engineering drawing. It must be:
  • Representative of the problem shape and relative proportions
  • Clean, uncluttered, and easy to read
  • Free of answer values or solution steps

ANTI-CHEATING: Label only what is GIVEN in the problem.
  • For Venn diagrams: circle labels (set names) only — no counts, no region values.
  • For geometry: given side lengths and angles only — mark unknowns with "?".
  • NEVER draw computed values, intermediate results, or the answer in the diagram.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY DIAGRAM WORKFLOW (when Requires_Diagram is true)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BEFORE writing any TikZ code, compute coordinates in the "diagram_precompute" field:

STEP A — Choose a scale so the largest dimension maps to ≤10 units.
STEP B — Compute ALL key coordinates from the given measurements:
  • Triangle vertices: place one at origin, one on x-axis, use Pythagorean/cosine rule for third.
  • Incircle center: I = (a·Ax + b·Bx + c·Cx)/(a+b+c), (a·Ay + b·By + c·Cy)/(a+b+c)
    where a,b,c are lengths of sides OPPOSITE to vertices A,B,C.
    Inradius r = Area / semi-perimeter.
  • Points on circles: verify distance from center = radius before placing them.
STEP C — Verify: list every constraint and confirm it holds numerically.

The TikZ code MUST use these pre-computed coordinates exactly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TikZ CODE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. \\documentclass[varwidth=21cm, border=5mm]{standalone}
2. \\usepackage{tikz} and explicitly load every library used.
3. fill=white on any node overlapping a line.
4. ALL raw coordinates strictly between -12 and +12.
5. No global \\scale transforms — use coordinate math instead.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a raw JSON object — no markdown fences, no preamble, no extra text after the closing brace.

{
  "id": "PLACEHOLDER",
  "text": "Question text. $...$ for inline math, $$...$$ for display.",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "correct_answer": "A",
  "explanation": "Complete step-by-step solution.",
  "Requires_Diagram": false,
  "diagram_precompute": null,
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
# 0b. MATH CRITIC PROMPT (Haiku)
# ==========================================
MATH_CRITIC_PROMPT = """You are a mathematics QA reviewer for competitive exam MCQ questions.

STEP 1 — CLASSIFY: QUANTITATIVE (needs calculation) / LOGICAL (reasoning only) / CONCEPTUAL (factual).

STEP 2 — VERIFY THE ANSWER independently:
  QUANTITATIVE: Solve from scratch, show every step. Does your answer match correct_answer exactly?
  LOGICAL: Apply pure logic. Does your conclusion match correct_answer?
  CONCEPTUAL: Verify factual accuracy.

STEP 3 — ANTI-CHEATING: Does the question text itself reveal the answer or any key intermediate result?
  (Do NOT look at TikZ code — that is reviewed separately.)

RESPONSE:
  Everything correct → reply with ONLY: PASS
  Any issue → numbered list. Always show your computed answer on math mismatches.
"""

# ==========================================
# 0c. DIAGRAM CRITIC PROMPT (Sonnet + Vision)
# FIX: Relaxed geometric tolerances — diagrams are visual aids, not CAD drawings.
# Advisory checks (G3-G5) do not block unless egregiously wrong.
# ==========================================
DIAGRAM_CRITIC_PROMPT = """You are a diagram QA reviewer with vision. You receive the rendered diagram image AND TikZ source code.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VISUAL CHECKS — from the rendered image (BLOCKING if failed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
V1. CLIPPING: Are any labels, vertices, or lines cut off at the image boundary?
V2. OVERLAPS: Do text labels overlap with lines or other labels making them unreadable?
V3. ANTI-CHEATING: Does the image show the answer value, any computed result,
    or solution steps? For Venn: are region counts shown? For geometry: is the unknown labelled?
V4. BASIC SHAPE CORRECTNESS: Does the shape look recognisable?
    A triangle should look like a triangle. A circle should look like a circle.
    This is a VISUAL AID for students — it does not need to be pixel-perfect,
    but it must not be completely misleading (e.g. a right triangle drawn as equilateral).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GEOMETRIC CHECKS — from TikZ code (BLOCKING only if severely wrong)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
G1. COORDINATE BOUNDS (BLOCKING): Any raw coordinate outside ±12? List violations.
G2. POINTS ON CIRCLES (BLOCKING if error > 15%):
    Compute distance from each claimed circle point to center. Compare to radius.
    Only fail if distance differs by more than 15% of the radius.
G3. TRIANGLE PROPORTIONS (ADVISORY — do NOT block on this alone):
    Check if drawn side ratios roughly match stated ratios. A 5:12:13 triangle
    should look visually distinct from an equilateral triangle.
    Flag only if proportions are so wrong the diagram would confuse students.
G4. INCIRCLE/CIRCUMCIRCLE (ADVISORY — do NOT block on this alone):
    Check center and radius are approximately correct. Tolerance ±15%.
    Minor inaccuracies here do not affect student understanding.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PASS if: No blocking failures in V1-V4 and G1-G2.
FAIL only if: At least one blocking check fails, OR G3/G4 is so egregiously wrong
              it would actively mislead a student (e.g. a right triangle looks obtuse).

RESPONSE:
  All checks pass → reply with ONLY: PASS
  Otherwise → numbered list of SPECIFIC blocking failures only.
  For geometric failures: show computed values vs drawn values.
  For visual failures: describe exactly what you see in the image.
  Keep it concise.
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
                f"Show new diagram_precompute. Return FULL JSON. Raw JSON only."
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
        # Keep diagram_precompute in raw_json_str for diagram critic, but strip from bank
        raw_for_critic = json.dumps(q_data)
        q_data.pop("diagram_precompute", None)
        return {
            "raw_json_str":     raw_for_critic,
            "question_data":    q_data,
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
    """Haiku — verifies answer correctness only. Does not see TikZ code."""
    q_data = state.get("question_data")
    print("\n🔢 [MathCritic/Haiku] Verifying answer...")

    if not q_data:
        return {
            "math_feedback":    "No question data.",
            "total_fail_count": state.get("total_fail_count", 0) + 1,
            "last_failure_type": "json",
        }

    # Strip TikZ — math critic doesn't need it (saves tokens, speeds up Haiku)
    q_for_critic = {k: v for k, v in q_data.items()
                    if k not in ("TikZ_Code", "diagram_precompute")}

    feedback = make_llm(_MODEL_HAIKU or _MODEL_SONNET, max_tokens=1024).invoke([
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
    Multimodal Sonnet — sees BOTH rendered PNG image AND TikZ source code.
    Visual checks: clipping, overlaps, anti-cheating, shape recognisability.
    Geometric checks: coordinate bounds, points on circles (relaxed tolerances).
    """
    q_data   = state.get("question_data")
    img_path = state.get("final_image_path")
    print("\n📐 [DiagramCritic/Sonnet+Vision] Verifying diagram...")

    if not needs_diagram(q_data):
        return {"diagram_feedback": None}

    # Retrieve precompute from raw_json_str (stripped from q_data)
    precompute = "Not provided"
    try:
        raw = json.loads(state.get("raw_json_str", "{}"))
        precompute = raw.get("diagram_precompute") or "Not provided"
    except Exception:
        pass

    text_prompt = (
        f"QUESTION TEXT:\n{q_data.get('text', '')}\n\n"
        f"CORRECT ANSWER: {q_data.get('correct_answer', '')} = "
        f"{q_data.get('options', {}).get(q_data.get('correct_answer', ''), '')}\n\n"
        f"TIKZ SOURCE:\n```latex\n{q_data.get('TikZ_Code', '')}\n```\n\n"
        f"COORDINATE PRE-COMPUTATION:\n{precompute}\n\n"
        f"Apply all visual checks (V1-V4) using the image and geometric "
        f"checks (G1-G4) using TikZ code and your own arithmetic."
    )

    # Build multimodal content
    png_b64 = None
    if img_path and os.path.exists(img_path):
        png_b64 = svg_to_png_base64(img_path)
        if png_b64:
            print("   🖼️  Image loaded for multimodal review")
        else:
            print("   ⚠️  SVG→PNG failed — text-only geometric review")
    else:
        print("   ⚠️  No rendered image — text-only geometric review")

    if png_b64:
        human_content = [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": png_b64
            }},
            {"type": "text", "text": text_prompt},
        ]
    else:
        human_content = text_prompt

    feedback = make_llm(_MODEL_SONNET, max_tokens=2048).invoke([
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
    print("\n🚀 Starting SSC CGL Seed Bank Pipeline...")
    print(f"   Haiku  : {_MODEL_HAIKU  or '⚠️  NOT SET — falls back to Sonnet'}")
    print(f"   Sonnet : {_MODEL_SONNET}")
    print(f"   Opus   : {_MODEL_OPUS}")
    print(f"   Critics: Math=Haiku | Diagram=Sonnet+Vision (multimodal)")
    print(f"   Budget : {MAX_RETRIES} attempts per round, infinite rounds per slot (never skips)")
    vision_status = "✅ cairosvg installed" if _CAIROSVG_AVAILABLE else "⚠️  cairosvg missing"
    print(f"   Vision : {vision_status}")

    output_file          = "ssc_cgl_question_bank.json"
    master_question_bank: List[Dict] = []

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            master_question_bank = json.load(f)

    used_numbers: List[str] = [
        fp for q in master_question_bank if (fp := numeric_fingerprint(q))
    ]

    # ── EDIT ONLY THIS LIST ─────────────────────────────────────────────────
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
    # ────────────────────────────────────────────────────────────────────────

    for subject, topic, sub_topic in TARGET_NODES:
        print(f"\n{'='*56}")
        print(f"🎯  {sub_topic}")
        print(f"{'='*56}")

        for difficulty in range(1, 6):
            for iteration in range(1, 3):
                print(f"\n👉  Level {difficulty} | Q {iteration}/2")

                slug = re.sub(r'[^A-Z0-9]', '', sub_topic.upper())[:10]

                # ── Infinite persistence loop ──────────────────────────────
                # Each iteration of this while-loop is one full LangGraph run
                # (MAX_RETRIES attempts). If the run exhausts all attempts without
                # banking, we immediately start a fresh run with a new ID and a
                # "fresh start" hint in the request — repeating until we bank.
                # The slot is NEVER skipped.
                round_num        = 0
                total_attempts   = 0
                banked           = False

                while not banked:
                    round_num += 1

                    # Fresh ID for each round so filenames don't collide
                    forced_id = f"SSC_{slug}_{difficulty}_{iteration}_{uuid.uuid4().hex[:6]}"

                    # On round 2+, tell the generator explicitly that previous
                    # attempts exhausted — approach the topic differently
                    extra_hint = ""
                    if round_num > 1:
                        extra_hint = (
                            f"\n- NOTE: Previous {round_num - 1} run(s) of "
                            f"{MAX_RETRIES} attempts each failed for this slot. "
                            f"Choose a COMPLETELY DIFFERENT concept, theorem, or "
                            f"problem type within the same subtopic and difficulty. "
                            f"Avoid anything tried before."
                        )

                    request = (
                        f"- Subject: {subject}\n"
                        f"- Topic: {topic}\n"
                        f"- Subtopic: {sub_topic}\n"
                        f"- Difficulty Level: {difficulty} / 5\n"
                        f"- Diagram_Mode: Auto (follow system prompt rules)\n"
                        + extra_hint
                    )

                    print(f"   🔁 Round {round_num} (slot will not be skipped)")

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

                    q_data = final_state.get("question_data")
                    succeeded = (
                        q_data
                        and not final_state.get("compile_error")
                        and not final_state.get("math_feedback")
                        and not final_state.get("diagram_feedback")
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
                            f"({used_in_run} attempts). Retrying with fresh concept..."
                        )


if __name__ == "__main__":
    run_seeder()