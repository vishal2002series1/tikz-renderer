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
PIVOT_AFTER_FAILS = 3
RENDERER_URL      = os.getenv("RENDERER_URL", "http://localhost:3002/api/render")

_MODEL_HAIKU  = os.getenv("Model_ID_Sonnet")                  # Haiku — fast, text-only questions
_MODEL_SONNET = os.getenv("Model_ID", "us.anthropic.claude-sonnet-4-6")
_MODEL_OPUS   = os.getenv("Model_ID_Opus", _MODEL_SONNET)     # fallback to Sonnet if Opus not set

# ==========================================
# 0a. GENERATOR SYSTEM PROMPT
# KEY FIX: mandatory coordinate pre-computation step before writing TikZ.
# This forces the LLM to verify geometric constraints BEFORE producing code.
# ==========================================
SYSTEM_PROMPT = """You are an expert exam question setter for competitive Indian exams and a LaTeX/TikZ expert.

Generate a single high-quality, exam-accurate MCQ question in the specified subject/topic.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIAGRAM DECISION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Only set "Requires_Diagram": true when a visual is NECESSARY to state the problem clearly.
Never produce a diagram if the question can be stated and solved without one.

ANTI-CHEATING: The diagram shows ONLY information given in the problem — never the answer,
never intermediate computed values, never solution steps.
For Venn diagrams: label circles with SET NAMES only.
For geometry: label only the GIVEN sides/angles. Mark unknowns with "?", not their value.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY DIAGRAM WORKFLOW (when Requires_Diagram is true)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You MUST follow these steps IN ORDER before writing any TikZ:

STEP A — CHOOSE SCALE: Pick a scale factor so all coordinates stay within ±12 units.
  Example: if a triangle has side 40 cm, scale = 0.25 so 40 × 0.25 = 10 ✓

STEP B — COMPUTE ALL COORDINATES from scratch using geometry:
  • For triangles: place one vertex at origin, one along x-axis, compute third vertex
    using actual side lengths and the cosine rule or Pythagorean theorem.
  • For circles: compute center and radius in SCALED units.
  • For incircles: center = (r_scaled, r_scaled) for right-angle-at-origin triangles,
    or use incenter formula: I = (a·A + b·B + c·C)/(a+b+c) for general triangles,
    where a,b,c are side lengths opposite to vertices A,B,C.
  • For circumcircles: center = circumcenter computed from perpendicular bisectors.

STEP C — VERIFY EVERY CONSTRAINT before writing code:
  • Each point claimed to be ON a circle: distance from center = radius ✓
  • Each tangent point: distance from incenter = inradius ✓  
  • Triangle side lengths match stated values (after scaling) ✓
  • All coordinates are strictly within ±12 ✓

STEP D — Write the TikZ code using the verified coordinates.

The computation in Steps A-C goes into a "diagram_precompute" field in your JSON.
This is required and must show actual numbers, not just descriptions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TikZ CODE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. \\documentclass[varwidth=21cm, border=5mm]{standalone}
2. \\usepackage{tikz} and explicitly load every library used.
3. Use fill=white on any node overlapping a line.
4. ALL raw coordinates strictly between -12 and +12 (not ±15).
   If you need larger values, apply a scale transformation.
5. Do NOT use \\scale or global transforms — use coordinate math instead.
6. Angles: use \\usetikzlibrary{angles,quotes} for angle arcs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a raw JSON object — no markdown fences, no preamble, no trailing text.

{
  "id": "PLACEHOLDER",
  "text": "Question text. Use $...$ for inline math, $$...$$ for display math.",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "correct_answer": "A",
  "explanation": "Complete step-by-step solution with all working shown.",
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

When Requires_Diagram is true, diagram_precompute MUST contain your coordinate
calculations (Steps A-C) as a text string showing all intermediate numbers.
"""

# ==========================================
# 0b. MATH CRITIC PROMPT (Haiku — fast)
# Only checks: answer correctness + answer leakage. Nothing else.
# ==========================================
MATH_CRITIC_PROMPT = """You are a mathematics QA reviewer for competitive exam questions.

You will receive an MCQ question as JSON. Your job is ONLY:

1. CLASSIFY: Is this QUANTITATIVE (needs calculation) or LOGICAL/CONCEPTUAL (no calculation)?

2. VERIFY THE ANSWER:
   - QUANTITATIVE: Solve from scratch. Show every step. 
     Does your answer exactly match correct_answer? If not, state the correct value.
   - LOGICAL: Apply pure logic independently. Does your conclusion match correct_answer?
   - CONCEPTUAL: Verify factual accuracy of correct_answer.

3. ANTI-CHEATING CHECK:
   Does the question text or diagram_precompute field contain the answer value or
   any intermediate computed result that would let a student skip the calculation?

DO NOT check diagram coordinates or TikZ code — that is handled separately.
DO NOT check visual layout or label positions.

RESPONSE:
  If answer is correct AND no cheating → reply with ONLY the single word: PASS
  Otherwise → short numbered list of specific issues. Always show your computed answer.
"""

# ==========================================
# 0c. DIAGRAM CRITIC PROMPT (Sonnet — can compute geometry)
# Only called when Requires_Diagram is true.
# Sonnet can actually compute distances, check circle constraints, verify proportions.
# ==========================================
DIAGRAM_CRITIC_PROMPT = """You are a geometric diagram QA reviewer with vision capabilities.
You will receive BOTH the rendered diagram image AND the TikZ source code simultaneously.
Use both inputs together — the image shows you what was actually rendered, the code shows
you the intended coordinates and structure.

PERFORM ALL OF THE FOLLOWING CHECKS:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VISUAL CHECKS (from the rendered image)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
V1. CLIPPING: Are any labels, points, or drawing elements clipped or cut off at the edges?
V2. OVERLAPS: Do any text labels overlap with lines, other labels, or geometric elements?
V3. PROPORTIONS: Does the visual shape look representative of the stated measurements?
    A triangle with sides 5:12:13 should look like a right triangle, not equilateral.
    A circle should look like a circle, not an ellipse.
V4. READABILITY: Are all labels legible? Are any too small or positioned confusingly?
V5. ANTI-CHEATING: Does the image show any computed answer values, intermediate results,
    or solution steps that would give away the answer to a student?
    For Venn diagrams: do any regions show counts or derived values?
    For geometry: is the unknown value labelled anywhere in the image?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GEOMETRIC ACCURACY CHECKS (from TikZ code + math)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
G1. COORDINATE BOUNDS: Are all raw coordinates strictly between -12 and +12?
G2. POINTS ON CIRCLES: For each point claimed to lie on a circle, compute:
    distance = sqrt((px-cx)² + (py-cy)²). Compare to circle radius. Tolerance ±0.05.
G3. TRIANGLE SIDE LENGTHS: Extract vertices, compute actual side lengths.
    Compare to question-stated lengths (accounting for any scale factor). Tolerance ±5%.
G4. INCIRCLE/CIRCUMCIRCLE PLACEMENT:
    Incircle: verify center = (a·A + b·B + c·C)/(a+b+c), radius = area/semi-perimeter.
    Circumcircle: verify all three vertices are equidistant from drawn center.
G5. TANGENT POINTS: Each tangent point should be the foot of perpendicular from
    incenter to the corresponding side.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If ALL checks pass → reply with ONLY: PASS
Otherwise → numbered list of specific failures.
For geometric failures, show your computed values vs drawn values.
For visual failures, describe exactly what you see in the image.
Be specific and concise.
"""

# ==========================================
# 0d. SVG → PNG HELPER for multimodal critic
# ==========================================
def svg_to_png_base64(svg_path: str, output_width: int = 900) -> Optional[str]:
    """
    Convert a rendered SVG file to a base64-encoded PNG string
    suitable for the Bedrock Claude multimodal API.
    Returns None if cairosvg is not installed or conversion fails.
    """
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
# ==========================================
class QuestionState(TypedDict):
    request_prompt:     str
    forced_id:          str
    generation_count:   int
    math_fail_count:    int   # consecutive math critic failures
    diagram_fail_count: int   # consecutive diagram critic failures
    raw_json_str:       Optional[str]
    question_data:      Optional[Dict[str, Any]]
    compile_error:      Optional[str]
    math_feedback:      Optional[str]   # from math critic
    diagram_feedback:   Optional[str]   # from diagram critic
    final_image_path:   Optional[str]
    used_numbers:       List[str]

# ==========================================
# 1a. HELPERS
# ==========================================
def extract_json(text: str) -> str:
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
    nums = sorted(set(re.findall(r'\b\d+(?:\.\d+)?\b', q_data.get("text", ""))))
    return ",".join(nums) if nums else ""


def is_pass(feedback: str) -> bool:
    """Robust PASS detection: first meaningful word is PASS."""
    words = feedback.strip().split()
    if not words:
        return False
    return words[0].upper().rstrip(".,!:*#") == "PASS"


def needs_diagram(q_data: Optional[dict]) -> bool:
    return bool(q_data and q_data.get("Requires_Diagram") and q_data.get("TikZ_Code"))


def pick_generator_model(gen_count: int, has_diagram_question: bool) -> tuple:
    """
    Smart model routing:
    - Questions needing diagrams start with Sonnet (Haiku cannot reliably compute geometry).
    - Text-only questions start with Haiku (fast, cheap, sufficient).
    - Both escalate to Opus after repeated failures.
    """
    if has_diagram_question:
        # Diagram questions: Sonnet from attempt 1, Opus from attempt 7
        if gen_count < 6:
            return _MODEL_SONNET, "Sonnet"
        else:
            return _MODEL_OPUS, "Opus" if _MODEL_OPUS != _MODEL_SONNET else "Sonnet"
    else:
        # Text-only questions: Haiku first, Sonnet after 3 fails, Opus after 6
        if gen_count < 3:
            model = _MODEL_HAIKU or _MODEL_SONNET
            label = "Haiku" if _MODEL_HAIKU else "Sonnet(fallback)"
        elif gen_count < 6:
            model, label = _MODEL_SONNET, "Sonnet"
        else:
            model = _MODEL_OPUS
            label = "Opus" if _MODEL_OPUS != _MODEL_SONNET else "Sonnet"
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
    math_fails        = state.get("math_fail_count", 0)
    diagram_fails     = state.get("diagram_fail_count", 0)
    total_fails       = math_fails + diagram_fails
    used_numbers      = state.get("used_numbers", [])

    # Determine if the current/previous question needs a diagram
    # for smarter model routing even before we know if the new one will
    prev_data         = state.get("question_data")
    prev_had_diagram  = needs_diagram(prev_data)

    model_id, model_label = pick_generator_model(gen_count, prev_had_diagram)
    print(f"\n🧠 [Generator/{model_label}] Attempt {gen_count + 1}...")

    # ── Base prompt ──
    prompt = (
        f"Generate an exam question for:\n"
        f"<request>\n{state['request_prompt']}\n</request>\n\n"
        f"Output ONLY raw JSON — no markdown fences, no preamble."
    )

    # ── Variety: ban used number sets ──
    if used_numbers:
        prompt += (
            "\n\nVARIETY: These number sets are already banked — use completely different numbers:\n"
            + "\n".join(f"  • {n}" for n in used_numbers[-8:])
        )

    prev_json = state.get("raw_json_str")

    # ── Mode: fix compile error ──
    if state.get("compile_error") and prev_json:
        print(f"   Mode: Fixing compile error")
        prompt += (
            f"\n\nPrevious JSON:\n```json\n{prev_json}\n```\n\n"
            f"TikZ failed to compile:\n<e>\n{state['compile_error']}\n</e>\n"
            f"Fix ONLY the TikZ. Re-run your coordinate verification (Steps A-C) first. "
            f"Return the FULL corrected JSON. Raw JSON only."
        )

    # ── Mode: fix diagram geometry ──
    elif state.get("diagram_feedback") and prev_json:
        if diagram_fails >= PIVOT_AFTER_FAILS:
            print(f"   Mode: Diagram pivot (fail #{diagram_fails})")
            prompt += (
                f"\n\nThe diagram has failed geometric QA {diagram_fails} times. "
                f"Generate a COMPLETELY DIFFERENT question that either:\n"
                f"  (a) Does not require a diagram, OR\n"
                f"  (b) Uses a simpler geometric configuration\n"
                f"Return fresh question as raw JSON only."
            )
        else:
            print(f"   Mode: Fixing diagram geometry (fail #{diagram_fails})")
            prompt += (
                f"\n\nPrevious JSON:\n```json\n{prev_json}\n```\n\n"
                f"A geometric diagram reviewer found these errors:\n"
                f"<diagram_feedback>\n{state['diagram_feedback']}\n</diagram_feedback>\n\n"
                f"You MUST redo the coordinate computation (Steps A-C) completely from scratch.\n"
                f"Do not patch existing coordinates — recompute everything.\n"
                f"Show your new diagram_precompute calculations.\n"
                f"Return the FULL corrected JSON. Raw JSON only."
            )

    # ── Mode: fix math/answer ──
    elif state.get("math_feedback") and prev_json:
        if math_fails >= PIVOT_AFTER_FAILS:
            print(f"   Mode: Math pivot (fail #{math_fails})")
            prompt += (
                f"\n\nThis question concept has failed math QA {math_fails} times. "
                f"Generate a COMPLETELY DIFFERENT question — different concept, different numbers.\n"
                f"Return fresh question as raw JSON only."
            )
        else:
            print(f"   Mode: Fixing math (fail #{math_fails})")
            prompt += (
                f"\n\nPrevious JSON:\n```json\n{prev_json}\n```\n\n"
                f"Math QA rejected it:\n<feedback>\n{state['math_feedback']}\n</feedback>\n\n"
                f"You MUST either:\n"
                f"  (a) Fix correct_answer AND options to match the reviewer's computed result, OR\n"
                f"  (b) Choose completely different numbers.\n"
                f"Do NOT reuse the same numbers — it will fail again.\n"
                f"Return the FULL corrected JSON. Raw JSON only."
            )

    llm = make_llm(model_id, max_tokens=8192)
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])

    raw = extract_json(response.content)
    try:
        q_data = json.loads(raw)
        q_data["id"] = state["forced_id"]
        # Remove diagram_precompute from stored JSON (internal scratch work, not for bank)
        q_data.pop("diagram_precompute", None)
        return {
            "raw_json_str":     json.dumps(q_data),
            "question_data":    q_data,
            "generation_count": gen_count + 1,
            "compile_error":    None,
            "math_feedback":    None,
            "diagram_feedback": None,
        }
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON parse failed: {e}")
        return {
            "question_data":    None,
            "compile_error":    f"Invalid JSON from LLM: {e}",
            "generation_count": gen_count + 1,
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
            return {"compile_error": err, "final_image_path": None}
    except Exception as e:
        print(f"   ❌ Renderer unreachable: {e}")
        return {"compile_error": f"Renderer unreachable: {e}", "final_image_path": None}


def math_critic_node(state: QuestionState) -> dict:
    """
    Fast Haiku critic — ONLY verifies answer correctness and anti-cheating.
    Does NOT look at TikZ code at all.
    """
    q_data = state.get("question_data")
    print("\n🔢 [MathCritic/Haiku] Verifying answer...")

    if not q_data:
        return {
            "math_feedback":  "No question data (JSON parse failed).",
            "math_fail_count": state.get("math_fail_count", 0) + 1,
        }

    # Strip TikZ from what we send to math critic — it doesn't need it and it's expensive tokens
    q_for_critic = {k: v for k, v in q_data.items() if k not in ("TikZ_Code", "diagram_precompute")}

    critic_prompt = (
        f"Review this exam question:\n\n"
        f"```json\n{json.dumps(q_for_critic, indent=2)}\n```"
    )

    critic_model = _MODEL_HAIKU or _MODEL_SONNET
    llm = make_llm(critic_model, max_tokens=1024)
    feedback = llm.invoke([
        SystemMessage(content=MATH_CRITIC_PROMPT),
        HumanMessage(content=critic_prompt),
    ]).content.strip()

    if is_pass(feedback):
        print("   ✅ Math approved!")
        return {"math_feedback": None, "math_fail_count": 0}
    else:
        fails = state.get("math_fail_count", 0) + 1
        print(f"   ⚠️  Math rejected (#{fails}): {feedback[:100]}...")
        return {"math_feedback": feedback, "math_fail_count": fails}


def diagram_critic_node(state: QuestionState) -> dict:
    """
    Multimodal Sonnet critic — receives BOTH the rendered PNG image AND the TikZ
    source code simultaneously.

    Visual checks (from image):  clipping, overlaps, proportions, readability,
                                  anti-cheating (answer visible in diagram?)
    Geometric checks (from code): coordinate bounds, points on circles,
                                   triangle side lengths, incircle/circumcircle accuracy.
    """
    q_data    = state.get("question_data")
    img_path  = state.get("final_image_path")

    print("\n📐 [DiagramCritic/Sonnet+Vision] Verifying diagram...")

    if not needs_diagram(q_data):
        return {"diagram_feedback": None, "diagram_fail_count": 0}

    # ── Build the text portion of the critic prompt ──
    text_prompt = (
        f"QUESTION TEXT:\n{q_data.get('text', '')}\n\n"
        f"CORRECT ANSWER: {q_data.get('correct_answer', '')} = "
        f"{q_data.get('options', {}).get(q_data.get('correct_answer', ''), '')}\n\n"
        f"TIKZ SOURCE CODE:\n```latex\n{q_data.get('TikZ_Code', '')}\n```\n\n"
        f"COORDINATE PRE-COMPUTATION (generator's own working):\n"
        f"{q_data.get('diagram_precompute', 'Not provided')}\n\n"
        f"Apply all visual checks (V1–V5) using the image above AND all geometric "
        f"checks (G1–G5) using the TikZ code and your own arithmetic."
    )

    # ── Try to load the rendered image as base64 PNG ──
    png_b64 = None
    if img_path and os.path.exists(img_path):
        png_b64 = svg_to_png_base64(img_path)
        if png_b64:
            print("   🖼️  Image loaded for multimodal review")
        else:
            print("   ⚠️  Could not convert SVG — falling back to text-only review")
    else:
        print("   ⚠️  No rendered image found — text-only geometric review")

    # ── Build the HumanMessage content ──
    # If we have the image, send it as the first content block (multimodal).
    # If not, send text only — critic still does coordinate math checks.
    if png_b64:
        human_content = [
            {
                "type": "image",
                "source": {
                    "type":       "base64",
                    "media_type": "image/png",
                    "data":       png_b64,
                },
            },
            {
                "type": "text",
                "text": text_prompt,
            },
        ]
    else:
        human_content = text_prompt   # plain string — text-only fallback

    llm = make_llm(_MODEL_SONNET, max_tokens=2048)
    feedback = llm.invoke([
        SystemMessage(content=DIAGRAM_CRITIC_PROMPT),
        HumanMessage(content=human_content),
    ]).content.strip()

    if is_pass(feedback):
        print("   ✅ Diagram approved (visual + geometric)!")
        return {"diagram_feedback": None, "diagram_fail_count": 0}
    else:
        fails = state.get("diagram_fail_count", 0) + 1
        print(f"   ⚠️  Diagram rejected (#{fails}): {feedback[:130]}...")
        return {"diagram_feedback": feedback, "diagram_fail_count": fails}

# ==========================================
# 3. ROUTING
# ==========================================
def route_after_compiler(state: QuestionState) -> str:
    if state.get("compile_error"):
        if state["generation_count"] >= MAX_RETRIES:
            print("🛑 Max retries on compile errors.")
            return END
        return "generator_node"
    if not state.get("question_data"):
        # JSON parse failed — go straight back to generator, skip both critics
        if state["generation_count"] >= MAX_RETRIES:
            return END
        return "generator_node"
    return "math_critic_node"


def route_after_math_critic(state: QuestionState) -> str:
    if state.get("math_feedback"):
        if state["generation_count"] >= MAX_RETRIES:
            print("🛑 Max retries on math feedback.")
            return END
        return "generator_node"
    # Math passed — now check diagram if one exists
    if needs_diagram(state.get("question_data")):
        return "diagram_critic_node"
    return END


def route_after_diagram_critic(state: QuestionState) -> str:
    if state.get("diagram_feedback"):
        if state["generation_count"] >= MAX_RETRIES:
            print("🛑 Max retries on diagram feedback.")
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
    print(f"   Haiku  : {_MODEL_HAIKU  or '⚠️  NOT SET — text-only Haiku falls back to Sonnet'}")
    print(f"   Sonnet : {_MODEL_SONNET}")
    print(f"   Opus   : {_MODEL_OPUS   or '⚠️  NOT SET — falls back to Sonnet'}")
    print(f"   Routing: text-only → Haiku→Sonnet→Opus | diagram → Sonnet→Opus")
    print(f"   Critics: Math=Haiku (fast) | Diagram=Sonnet+Vision (image + TikZ code)")
    vision_status = "✅ cairosvg installed" if _CAIROSVG_AVAILABLE else "⚠️  cairosvg missing — install with: pip install cairosvg --break-system-packages"
    print(f"   Vision : {vision_status}")

    output_file          = "ssc_cgl_question_bank.json"
    master_question_bank: List[Dict] = []

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            master_question_bank = json.load(f)

    used_numbers: List[str] = [
        fp for q in master_question_bank
        if (fp := numeric_fingerprint(q))
    ]

    # ── EDIT ONLY THIS LIST ────────────────────────────────────────────────
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
        print(f"\n{'='*54}")
        print(f"🎯  {sub_topic}")
        print(f"{'='*54}")

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
                    "request_prompt":     request,
                    "forced_id":          forced_id,
                    "generation_count":   0,
                    "math_fail_count":    0,
                    "diagram_fail_count": 0,
                    "raw_json_str":       None,
                    "question_data":      None,
                    "compile_error":      None,
                    "math_feedback":      None,
                    "diagram_feedback":   None,
                    "final_image_path":   None,
                    "used_numbers":       list(used_numbers),
                }

                final_state = app.invoke(initial_state)

                q_data    = final_state.get("question_data")
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
                    had_diagram = "📐" if q_data.get("Requires_Diagram") else "📝"
                    print(f"   💾 Banked {had_diagram}: {q_data['id']}")
                else:
                    print(f"   🛑 Failed after {final_state.get('generation_count', 0)} attempts.")


if __name__ == "__main__":
    run_seeder()