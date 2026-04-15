"""
generic_exam_question_bank.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generic multi-exam question bank generator.
Supports any exam in syllabus_maps.json — question style, difficulty scale,
LaTeX conventions, and critic logic all adapt automatically to the chosen exam.

CONFIGURABLE PARAMETERS (edit the RUN CONFIG section at the bottom):
  EXAM          – exact key from syllabus_maps.json (e.g. "AWS Solutions Architect Associate")
  SUBJECT       – subject/domain name, or "All"
  TOPIC         – topic name, or "All"
  SUB_TOPIC     – subtopic name, or "All"
  N_PER_LEVEL   – questions to bank per difficulty level per iteration (default 2)
  K_ITERATIONS  – iterations — K=2 doubles the total questions (default 1)
  DIFFICULTY_LEVELS – list of difficulty levels to generate (default [1,2,3,4,5])
  SYLLABUS_FILE – path to syllabus_maps.json
  OUTPUT_FILE   – output JSON path (set explicitly per exam to avoid cross-exam collisions)
"""

import os
import re
import json
import uuid
import base64
import requests
from typing import TypedDict, Optional, Dict, Any, List, Tuple
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

_MODEL_HAIKU  = os.getenv("Model_ID_Sonnet")
_MODEL_SONNET = os.getenv("Model_ID", "us.anthropic.claude-sonnet-4-6")
_MODEL_OPUS   = os.getenv("Model_ID_Opus", _MODEL_SONNET)

# ==========================================
# 0a. EXAM CATEGORY DETECTION
# Maps any exam name to a category that drives prompt selection.
# ==========================================
def _exam_category(exam: str) -> str:
    """
    Returns a short category string for the given exam name.
    Add new exams here as the syllabus grows.
    """
    e = exam.lower()
    if "upsc" in e or "civil services" in e or "ias" in e:
        return "upsc_gs"
    if "ssc" in e or "cgl" in e or "chsl" in e:
        return "ssc"
    if "ibps" in e or "rrb" in e or "rbi" in e or "bank" in e:
        return "banking"
    if "gate" in e:
        return "gate"
    if "aws" in e or "azure" in e or "gcp" in e or "google cloud" in e or "cloud" in e:
        return "cloud_cert"
    if "lean six sigma" in e or "lssbb" in e or "iassc" in e:
        return "lssbb"
    if "pmp" in e or "project management" in e:
        return "pmp"
    if "power bi" in e or "pl-300" in e:
        return "powerbi"
    return "generic"


# ==========================================
# 0b. DYNAMIC PROMPT BUILDERS
# ==========================================

def build_system_prompt(exam: str) -> str:
    """
    Returns a SYSTEM_PROMPT string tailored to the exam category.
    All prompts share the same TikZ rules and output format skeleton;
    only the exam-specific sections differ.
    """
    cat = _exam_category(exam)

    # ── Per-category question-style block ──────────────────────────────────
    if cat == "upsc_gs":
        style_block = f"""You are an expert {exam} question setter and a LaTeX/TikZ expert.

Generate a single high-quality, exam-accurate MCQ for {exam}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAM STYLE & PATTERNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UPSC Prelims GS questions are 100% conceptual and factual — NO numerical calculations.
All questions are 4-option MCQs with a single correct answer.

Authentic UPSC question patterns (use these):
  STATEMENT TYPE:
    "Consider the following statements:
     1. ...  2. ...
     Which of the above statements is/are correct?
     (a) 1 only  (b) 2 only  (c) Both 1 and 2  (d) Neither 1 nor 2"
  MATCHING TYPE: "Which of the following pairs is/are correctly matched?"
  DIRECT FACTUAL: "With reference to [topic], which of the following is correct?"
  ASSERTION-REASON:
    "Assertion (A): ...  Reason (R): ...
     (a) Both A and R are true and R is the correct explanation of A
     (b) Both A and R are true but R is NOT the correct explanation of A
     (c) A is true but R is false  (d) A is false but R is true"

Use Statement-type most frequently (most common UPSC pattern).
Wrong options must exploit common misconceptions. No vague language (sometimes/generally/may).
Constitutional articles, case names, dates, and convention years must be exact.

Difficulty scale:
  1 = Direct recall of a single fact
  2 = Distinguishing between similar concepts
  3 = Exceptions, nuances, or landmark cases
  4 = Statement-based with 2-3 statements requiring careful elimination
  5 = Complex multi-statement or assertion-reason requiring deep understanding

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LaTeX FORMATTING — MANDATORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES:
  \\textbf{{Article 21}}, \\textbf{{Article 32}}   — article references
  \\textit{{Kesavananda Bharati v. State of Kerala (1973)}}  — case citations
  \\textbf{{Statement 1:}} ... in text field
  \\textbf{{Statement 1 — TRUE:}} / \\textbf{{Statement 1 — FALSE:}} in explanation
  \\textbf{{Assertion (A):}}  and  \\textbf{{Reason (R):}}
  \\textbf{{1 and 2 only}},  \\textbf{{1, 2 and 3}}  — combo options

WORKED EXAMPLE — Statement-based:
  "text": "...\\\\textbf{{Statement 1:}} \\\\textbf{{Article 14}} guarantees equality...\\nWhich is/are correct?",
  "options": {{"A": "\\\\textbf{{1 and 2 only}}", "B": "\\\\textbf{{2 and 3 only}}", "C": "\\\\textbf{{1 only}}", "D": "\\\\textbf{{1, 2 and 3}}"}},
  "explanation": "\\\\textbf{{Statement 1 — TRUE:}} \\\\textbf{{Article 14}} uses 'any person'...\\n\\n\\\\textbf{{Statement 2 — FALSE:}} ..."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GROUND TRUTH ANCHORS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Biodiversity Hotspots (CI): $\\geq$1,500 endemic plants; LOST $\\geq$70% primary vegetation ($\\leq$30% remains); 36 hotspots; India's 4: Western Ghats & Sri Lanka, Himalaya, Indo-Burma, Sundaland.
IUCN: EX→EW→CR(80%)→EN(50%)→VU(30%)→NT→LC→DD→NE  (% = population reduction criterion A)
Articles: 12(State def), 13(void laws), 14(equality/all persons), 15(no disc/citizens), 16(4)(appointments), 16(4A)(promotions/77th Amdt), 21(life & liberty), 21A(education/86th Amdt), 32(move SC=FR itself; suspended under Art 359 not 352), 352/356/360(emergencies), 359(suspend FR enforcement)."""

    elif cat == "cloud_cert":
        style_block = f"""You are an expert {exam} question setter and a LaTeX/TikZ expert.

Generate a single high-quality, exam-accurate MCQ for {exam}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAM STYLE & PATTERNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{exam} questions are scenario-based, application-focused MCQs.
All questions are 4-option single-answer MCQs unless noted.

Draw on your knowledge of REAL exam question styles:
  SCENARIO TYPE (most common ~60%):
    "A company needs to [business requirement]. The solution must [constraint].
     Which [service/approach/configuration] meets these requirements?"
  BEST-PRACTICE TYPE (~20%):
    "Which of the following is the MOST [cost-effective / secure / operationally efficient] way to...?"
  TROUBLESHOOTING TYPE (~10%):
    "A solutions architect is reviewing an architecture [with a described problem]. 
     What change should be made?"
  CONCEPTUAL TYPE (~10%):
    "Which statement BEST describes [service/feature/concept]?"

Question-writing rules:
  • Every scenario MUST specify a clear business or technical requirement.
  • Include relevant constraints (cost, ops overhead, performance, security, latency).
  • Wrong options must be plausible but fail on one specific constraint in the scenario.
  • Never reveal the answer in the question stem.
  • Service names in \\textbf{{}} — e.g. \\textbf{{Amazon S3}}, \\textbf{{AWS IAM}}, \\textbf{{Amazon RDS}}.

Difficulty scale (SAA-C03 Associate level):
  1 = Identify the correct service for a single stated requirement
  2 = Choose between 2 similar services given a specific constraint
  3 = Multi-constraint scenario (cost + performance, or security + availability)
  4 = Architect-level trade-off analysis across 2-3 services/features
  5 = Complex multi-service solution design with operational and cost optimisation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LaTeX FORMATTING — MANDATORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES:
  \\textbf{{Amazon S3}}, \\textbf{{AWS IAM}}, \\textbf{{Amazon RDS}}  — all AWS service names
  \\textbf{{IAM Role}}, \\textbf{{Security Group}}, \\textbf{{VPC}}   — key technical terms
  \\textit{{least privilege}}, \\textit{{shared responsibility model}}  — principle names
  \\textbf{{MOST cost-effective}}, \\textbf{{LEAST operational overhead}}  — emphasis words in options

WORKED EXAMPLE:
  "text": "A company stores customer data in \\\\textbf{{Amazon S3}}. The security team requires that all data be encrypted at rest and that encryption keys be rotated annually without application changes. Which solution meets these requirements with the \\\\textbf{{LEAST operational overhead}}?",
  "options": {{
    "A": "Use \\\\textbf{{SSE-S3}} (\\\\textit{{Server-Side Encryption with S3-managed keys}})",
    "B": "Use \\\\textbf{{SSE-KMS}} with an \\\\textbf{{AWS managed key}} and enable automatic key rotation",
    "C": "Use \\\\textbf{{SSE-C}} with customer-provided keys rotated by a Lambda function",
    "D": "Use client-side encryption with a custom key management solution"
  }},
  "explanation": "\\\\textbf{{B is correct:}} \\\\textbf{{SSE-KMS}} with an AWS managed key supports automatic annual rotation natively — no application changes required...\\n\\n\\\\textbf{{A is incorrect:}} \\\\textbf{{SSE-S3}} encrypts data but S3-managed keys cannot be configured for custom rotation schedules..."
"""

    elif cat == "lssbb":
        style_block = f"""You are an expert {exam} question setter and a LaTeX/TikZ expert.

Generate a single high-quality, exam-accurate MCQ for {exam}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAM STYLE & PATTERNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IASSC LSSBB questions test the full DMAIC Body of Knowledge.
Mix of conceptual, calculation, and scenario-based question types.

Draw on your knowledge of real IASSC exam styles:
  DEFINITION/CONCEPT TYPE: "Which of the following BEST describes [term/tool]?"
  CALCULATION TYPE: "A process has USL=50, LSL=30, μ=40, σ=3. What is the Cpk?"
  APPLICATION TYPE: "A Black Belt notices [symptom]. Which tool should be used FIRST?"
  SCENARIO TYPE: "During the [phase], a team finds [data]. What does this indicate?"

Key formulas to use exactly:
  Cp=(USL-LSL)/(6σ), Cpk=min[(USL-μ)/3σ, (μ-LSL)/3σ]
  Pp=(USL-LSL)/(6s), Ppk=min[(USL-μ)/3s, (μ-LSL)/3s]
  DPMO=(Defects/(Units×Opps))×1,000,000, DPU=D/U, RTY=e^(-total DPU)
  Type I error=α=reject true H₀, Type II error=β=fail to reject false H₀, Power=1-β

Difficulty scale:
  1 = Recall a definition or formula
  2 = Single-step calculation or direct concept application
  3 = Multi-step calculation or concept comparison (Cpk vs Ppk)
  4 = Scenario analysis requiring DMAIC phase knowledge
  5 = Complex scenario with tool selection and interpretation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LaTeX FORMATTING — MANDATORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  \\textbf{{Cpk}}, \\textbf{{DPMO}}, \\textbf{{RTY}}  — metric names
  $C_{{pk}} = \\min\\left[\\frac{{USL-\\mu}}{{3\\sigma}}, \\frac{{\\mu-LSL}}{{3\\sigma}}\\right]$  — formulas inline
  \\textbf{{Measure Phase}}, \\textbf{{Analyse Phase}}  — DMAIC phases
  \\textbf{{Statement 1 — TRUE:}} / \\textbf{{Statement 1 — FALSE:}}  — in explanations
"""

    elif cat in ("ssc", "banking", "gate"):
        style_block = f"""You are an expert {exam} question setter and a LaTeX/TikZ expert.

Generate a single high-quality, exam-accurate MCQ for {exam}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAM STYLE & PATTERNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Draw on your knowledge of real {exam} question patterns and difficulty distribution.
Use the question style that authentically mirrors this exam's paper format.
Wrong options must be based on common errors, not obviously silly distractors.

Difficulty scale:
  1 = Direct recall or single-step  2 = Two-step or concept distinction
  3 = Multi-step or application     4 = Complex application or reasoning
  5 = Hardest exam-level question

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LaTeX FORMATTING — MANDATORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  $...$ for inline math, $$...$$ for display math.
  \\textbf{{key term}} for important concepts and terms.
  \\textbf{{Statement N — TRUE/FALSE:}} structure in explanations.
"""

    else:  # generic / pmp / powerbi / etc.
        style_block = f"""You are an expert {exam} question setter and a LaTeX/TikZ expert.

Generate a single high-quality, exam-accurate MCQ for {exam}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAM STYLE & PATTERNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Draw extensively on your knowledge of how {exam} questions are actually structured and worded in real exams.
Mirror the authentic question style, terminology, and difficulty distribution of this certification.
Wrong options must be plausible and representative of real exam distractors.

Difficulty scale:
  1 = Core knowledge recall  2 = Concept application
  3 = Multi-concept analysis  4 = Scenario / case study
  5 = Complex judgment / best-practice decision

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LaTeX FORMATTING — MANDATORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  \\textbf{{key term}} for important concepts, tools, and service names.
  \\textit{{principle or methodology name}} for named principles.
  $...$ for any mathematical notation.
  \\textbf{{Statement N — TRUE/FALSE:}} in explanation headers.
"""

    # ── Shared tail: diagram rules + output format (same for all exams) ────
    shared_tail = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIAGRAM DECISION — MANDATORY REASONING STEP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before outputting JSON, ask: "Would a student be UNABLE to understand this question without a figure?"

Set "Requires_Diagram": true ONLY when a visual is genuinely necessary — architecture diagrams,
network topology, geographic maps, process flowcharts, statistical charts.
Set "Requires_Diagram": false (default) for purely textual/conceptual questions.
When in doubt → false.

When a diagram IS needed:
  • TikZ only. Keep it extremely simple — labels, boxes, arrows.
  • ANTI-CHEATING: never draw the answer or any computed value.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TikZ CODE RULES (when diagram is needed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. \\documentclass[varwidth=21cm, border=5mm]{standalone}
2. \\usepackage{tikz} + explicitly load every library used.
3. fill=white on any node overlapping a line.
4. ALL raw coordinates strictly between -12 and +12.
5. No global \\scale transforms. Keep it SIMPLE.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a raw JSON object — no markdown fences, no preamble, no text after closing brace.
Every string field MUST contain LaTeX formatting.

{
  "id": "PLACEHOLDER",
  "text": "Question text with \\\\textbf{} and \\\\textit{} LaTeX.",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "correct_answer": "A",
  "explanation": "Structured explanation with \\\\textbf{key term} formatting.",
  "Requires_Diagram": false,
  "TikZ_Code": null,
  "metadata": {
    "exam": "",
    "subject": "",
    "topic": "",
    "sub_topic": "",
    "difficulty_level": 1
  }
}
"""
    return style_block + shared_tail


def build_critic_prompt(exam: str) -> str:
    """
    Returns a MATH_CRITIC_PROMPT tailored to the exam category.
    The critic is told exactly which exam it is reviewing so it never
    rejects valid questions as 'wrong exam context'.
    """
    cat = _exam_category(exam)

    if cat == "upsc_gs":
        domain_block = f"""You are a strict QA reviewer for {exam} questions.
You have deep expertise in Indian Polity, History, Geography, Environment, Economy, and General Science as tested in UPSC Prelims.
{exam} has NO numerical calculations — all questions are conceptual and factual.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GROUND TRUTH — USE THESE FACTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Biodiversity Hotspots (CI): ≥1,500 endemic plants; LOST ≥70% primary vegetation (≤30% remains); 36 hotspots; India: Western Ghats & Sri Lanka, Himalaya, Indo-Burma, Sundaland.
IUCN: EX→EW→CR(≥80%)→EN(≥50%)→VU(≥30%)→NT→LC→DD→NE
Key Articles: 12(State), 13(void), 14(equality/all persons), 15(no-disc/citizens), 16(4)(appts), 16(4A)(promotions/77th), 21(life), 21A(education/86th), 32(FR/can suspend Art 359), 352/356/360(emergencies), 359(suspend FR enforcement).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REVIEW STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — CLASSIFY: DIRECT FACTUAL / STATEMENT-BASED / MATCHING / ASSERTION-REASON
STEP 2 — VERIFY FACTUAL ACCURACY:
  • STATEMENT-BASED: verify EACH statement individually (T/F). Does your T/F pattern match correct_answer?
  • MATCHING: verify every pair independently.
  • ASSERTION-REASON: verify A and R independently; does R correctly explain A?
  • Articles must be precisely correct (Art 21 vs 21A, Art 352 vs 359).
  • Case names and years must be exact.
STEP 3 — OPTION QUALITY: all 4 options plausible; correct answer unambiguous.
STEP 4 — LaTeX CHECK (reject if missing): \\textbf{{Article XX}} in text; \\textbf{{Statement N — TRUE/FALSE:}} in explanation; \\textbf{{combo}} in options. If ALL absent → flag "LaTeX missing".
STEP 5 — STYLE: no "sometimes/generally/may"; question doesn't reveal answer."""

    elif cat == "cloud_cert":
        domain_block = f"""You are a strict QA reviewer for {exam} questions.
You have deep expertise in AWS/cloud services, architecture best practices, and the exam's body of knowledge.
You are reviewing questions for {exam} — this is the CORRECT exam context.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REVIEW STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — CLASSIFY: SCENARIO / BEST-PRACTICE / TROUBLESHOOTING / CONCEPTUAL
STEP 2 — VERIFY TECHNICAL ACCURACY:
  • Is the correct_answer technically correct for the stated scenario and constraints?
  • Verify each wrong option: does it fail the scenario for a clear, defensible reason?
  • Service names, limits, and behaviours must be accurate (e.g. S3 event notifications, RDS Multi-AZ sync replication, SQS visibility timeout).
  • For IAM questions: verify policy evaluation logic, permission boundaries, SCPs carefully.
  • Pricing/cost claims must be directionally correct (not precise numbers).
STEP 3 — SCENARIO QUALITY: is there a clear business/technical requirement? Are constraints specified? Would only one option satisfy ALL constraints?
STEP 4 — LaTeX CHECK: \\textbf{{Service names}} in text; key emphasis words bold in options. Flag if all LaTeX absent.
STEP 5 — ANTI-CHEATING: question stem doesn't reveal the answer."""

    elif cat == "lssbb":
        domain_block = f"""You are a strict QA reviewer for {exam} questions.
You have deep expertise in Six Sigma statistics, Lean tools, DMAIC, and the IASSC Body of Knowledge.
You are reviewing questions for {exam} — this is the CORRECT exam context.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REVIEW STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — CLASSIFY: DEFINITION / CALCULATION / APPLICATION / SCENARIO
STEP 2 — VERIFY ACCURACY:
  • CALCULATION: apply correct IASSC formula; show steps; does result match correct_answer?
  • DEFINITION: is correct_answer accurate per IASSC BoK?
  • Terminology: Cpk vs Ppk (σ vs s), Type I=α=reject true H₀, Type II=β.
STEP 3 — OPTION QUALITY: all 4 plausible; correct answer unambiguous.
STEP 4 — LaTeX CHECK: formulas use $...$ notation; key terms \\textbf{{}}. Flag if all absent.
STEP 5 — ANTI-CHEATING: question doesn't reveal answer."""

    else:  # generic / ssc / banking / gate / pmp / powerbi
        domain_block = f"""You are a strict QA reviewer for {exam} questions.
You have deep expertise in {exam} and its body of knowledge.
You are reviewing questions for {exam} — this is the CORRECT exam context.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REVIEW STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — CLASSIFY the question type.
STEP 2 — VERIFY ACCURACY: Is the correct_answer factually/technically correct?
  • For calculation questions: compute independently and verify.
  • For conceptual questions: verify against {exam} body of knowledge.
  • Verify each wrong option is definitively incorrect.
STEP 3 — OPTION QUALITY: all 4 options plausible; correct answer unambiguous.
STEP 4 — LaTeX CHECK: key terms use \\textbf{{}}; formulas use $...$. Flag if all absent.
STEP 5 — ANTI-CHEATING: question stem doesn't reveal the answer."""

    shared_response = """

RESPONSE:
  Everything correct → reply with ONLY the single word: PASS
  Any issue → short numbered list. State the correct fact/value for every error found.
  For statement-based errors: state which statements are actually T/F and why.
"""
    return domain_block + shared_response

# ==========================================
# 0c. DIAGRAM CRITIC PROMPT (Sonnet + Vision)
# ==========================================
DIAGRAM_CRITIC_PROMPT = """You are a diagram visual QA reviewer. You receive the rendered diagram image.
Your job is purely visual — you are NOT checking mathematical accuracy (that is done separately).

Look at the image and check ONLY these four things:

V1. CLIPPING: Are any labels, lines, or shapes cut off at the image boundary?
    Fail if any element is partially outside the frame.

V2. OVERLAPS: Do text labels overlap with lines or other labels making them unreadable?
    Fail if any label is obscured or illegible.

V3. ANTI-CHEATING: Does the diagram show the answer value or any computed result?
    Fail if the answer or any intermediate computed value appears in the diagram.

V4. BASIC RECOGNISABILITY: Is the main shape recognisable?
    Fail ONLY if the shape is so distorted it would genuinely confuse a student.

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
# ==========================================
class QuestionState(TypedDict):
    request_prompt:    str
    forced_id:         str
    system_prompt:     str          # built dynamically per exam at run time
    critic_prompt:     str          # built dynamically per exam at run time
    generation_count:  int
    total_fail_count:  int
    last_failure_type: str
    raw_json_str:      Optional[str]
    question_data:     Optional[Dict[str, Any]]
    compile_error:     Optional[str]
    math_feedback:     Optional[str]
    diagram_feedback:  Optional[str]
    final_image_path:  Optional[str]
    used_numbers:      List[str]

# ==========================================
# 1a. HELPERS
# ==========================================
def extract_json(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?[ \t]*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    first_brace = text.find('{')
    last_brace  = text.rfind('}')
    if first_brace != -1 and last_brace > first_brace:
        return text[first_brace:last_brace + 1]
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("\n", 1)[0]
    return text.strip()


def numeric_fingerprint(q_data: dict) -> str:
    nums = sorted(set(re.findall(r'\b\d+(?:\.\d+)?\b', q_data.get("text", ""))))
    return ",".join(nums) if nums else ""


def is_pass(feedback: str) -> bool:
    words = feedback.strip().split()
    return bool(words) and words[0].upper().rstrip(".,!:*#") == "PASS"


def needs_diagram(q_data: Optional[dict]) -> bool:
    return bool(q_data and q_data.get("Requires_Diagram") and q_data.get("TikZ_Code"))


def pick_generator_model(gen_count: int, has_diagram: bool) -> tuple:
    if has_diagram:
        return (_MODEL_SONNET, "Sonnet") if gen_count < 6 else (
            _MODEL_OPUS, "Opus" if _MODEL_OPUS != _MODEL_SONNET else "Sonnet")
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
# 1b. SYLLABUS RESOLVER
# Converts (exam, subject, topic, sub_topic) with "All" wildcards
# into a flat list of (subject, topic, sub_topic) tuples to iterate over.
# Works with both flat (subject > [subtopics]) and nested (subject > topic > [subtopics]).
# ==========================================
def resolve_target_nodes(
    syllabus: dict,
    subject:  str = "All",
    topic:    str = "All",
    sub_topic: str = "All",
) -> List[Tuple[str, str, str]]:
    """
    Returns a list of (subject, topic, sub_topic) tuples.
    topic is set to "" when the syllabus is flat (subject > [subtopics]).
    """
    nodes: List[Tuple[str, str, str]] = []

    for subj_name, subj_val in syllabus.items():
        if subject != "All" and subj_name != subject:
            continue

        if isinstance(subj_val, list):
            # Flat structure: subject > [subtopics]  (e.g. UPSC GS-1)
            for st in subj_val:
                if sub_topic == "All" or st == sub_topic:
                    nodes.append((subj_name, "", st))

        elif isinstance(subj_val, dict):
            # Nested structure: subject > topic > [subtopics]  (e.g. IASSC, SSC)
            for topic_name, topic_val in subj_val.items():
                if topic != "All" and topic_name != topic:
                    continue
                subtopics = topic_val if isinstance(topic_val, list) else [topic_val]
                for st in subtopics:
                    if sub_topic == "All" or st == sub_topic:
                        nodes.append((subj_name, topic_name, st))

    return nodes

# ==========================================
# 2. GRAPH NODES (identical architecture to SSC/IASSC files)
# ==========================================

def generator_node(state: QuestionState) -> dict:
    gen_count        = state.get("generation_count", 0)
    total_fails      = state.get("total_fail_count", 0)
    last_failure     = state.get("last_failure_type", "")
    used_numbers     = state.get("used_numbers", [])
    prev_had_diagram = needs_diagram(state.get("question_data"))
    sys_prompt       = state.get("system_prompt", "")

    model_id, model_label = pick_generator_model(gen_count, prev_had_diagram)
    print(f"\n🧠 [Generator/{model_label}] Attempt {gen_count + 1}...")

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

    if last_failure == "compile" and prev_json:
        print("   Mode: Fixing compile error")
        prompt += (
            f"\n\nPrevious JSON:\n```json\n{prev_json}\n```\n\n"
            f"TikZ failed to compile:\n<e>\n{state['compile_error']}\n</e>\n"
            f"Fix ONLY the TikZ. Return FULL corrected JSON. Raw JSON only."
        )
    elif last_failure == "diagram" and prev_json:
        if total_fails >= PIVOT_AFTER_FAILS:
            print(f"   Mode: Diagram pivot (total fails: {total_fails})")
            prompt += (
                f"\n\nDiagram QA failed {total_fails} times. "
                f"Generate a COMPLETELY DIFFERENT question — text-only preferred. Raw JSON only."
            )
        else:
            print(f"   Mode: Fixing diagram only (total fails: {total_fails})")
            prompt += (
                f"\n\nPrevious JSON:\n```json\n{prev_json}\n```\n\n"
                f"Diagram has visual errors:\n"
                f"<diagram_feedback>\n{state['diagram_feedback']}\n</diagram_feedback>\n\n"
                f"Keep question text/options/answer EXACTLY the same. ONLY fix TikZ_Code. "
                f"Return FULL JSON. Raw JSON only."
            )
    elif last_failure == "math" and prev_json:
        if total_fails >= PIVOT_AFTER_FAILS:
            print(f"   Mode: Pivot (total fails: {total_fails})")
            prompt += (
                f"\n\nThis question concept failed QA {total_fails} times. "
                f"Generate a COMPLETELY DIFFERENT question — different concept, service, or angle on the sub-topic. "
                f"Raw JSON only."
            )
        else:
            print(f"   Mode: Fixing errors (total fails: {total_fails})")
            prompt += (
                f"\n\nPrevious JSON:\n```json\n{prev_json}\n```\n\n"
                f"QA Reviewer rejected it:\n<feedback>\n{state['math_feedback']}\n</feedback>\n\n"
                f"Fix the errors exactly as described. "
                f"Raw JSON only, nothing after closing brace."
            )

    llm = make_llm(model_id, max_tokens=8192)
    response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=prompt)])

    raw = extract_json(response.content)
    try:
        q_data = json.loads(raw)
        q_data["id"] = state["forced_id"]
        return {
            "raw_json_str":      json.dumps(q_data),
            "question_data":     q_data,
            "generation_count":  gen_count + 1,
            "compile_error":     None,
            "math_feedback":     None,
            "diagram_feedback":  None,
            "last_failure_type": "",
        }
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON parse failed: {e}")
        return {
            "question_data":     None,
            "compile_error":     f"JSON parse error: {e}",
            "generation_count":  gen_count + 1,
            "total_fail_count":  state.get("total_fail_count", 0) + 1,
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
            gen      = state.get("generation_count", 0)
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
                "compile_error":     err,
                "final_image_path":  None,
                "total_fail_count":  state.get("total_fail_count", 0) + 1,
                "last_failure_type": "compile",
            }
    except Exception as e:
        print(f"   ❌ Renderer unreachable: {e}")
        return {
            "compile_error":     str(e),
            "final_image_path":  None,
            "total_fail_count":  state.get("total_fail_count", 0) + 1,
            "last_failure_type": "compile",
        }


def math_critic_node(state: QuestionState) -> dict:
    q_data      = state.get("question_data")
    crit_prompt = state.get("critic_prompt", "")
    print("\n🔢 [FactualCritic/Sonnet] Verifying...")

    if not q_data:
        return {
            "math_feedback":     "No question data.",
            "total_fail_count":  state.get("total_fail_count", 0) + 1,
            "last_failure_type": "json",
        }

    q_for_critic = {k: v for k, v in q_data.items() if k != "TikZ_Code"}
    feedback = make_llm(_MODEL_SONNET, max_tokens=1024).invoke([
        SystemMessage(content=crit_prompt),
        HumanMessage(content=f"Review:\n```json\n{json.dumps(q_for_critic, indent=2)}\n```"),
    ]).content.strip()

    if is_pass(feedback):
        print("   ✅ Approved!")
        return {"math_feedback": None}
    else:
        fails = state.get("total_fail_count", 0) + 1
        print(f"   ⚠️  Rejected (total fails: {fails}): {feedback}...")
        return {
            "math_feedback":     feedback,
            "total_fail_count":  fails,
            "last_failure_type": "math",
        }


def diagram_critic_node(state: QuestionState) -> dict:
    q_data   = state.get("question_data")
    img_path = state.get("final_image_path")
    print("\n📐 [DiagramCritic/Sonnet+Vision] Visual check...")

    if not needs_diagram(q_data):
        return {"diagram_feedback": None}

    png_b64 = None
    if img_path and os.path.exists(img_path):
        png_b64 = svg_to_png_base64(img_path)
        if png_b64:
            print("   🖼️  Image loaded for visual review")

    if not png_b64:
        print("   ✅ Diagram check skipped (no image available)")
        return {"diagram_feedback": None}

    human_content = [
        {"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": png_b64
        }},
        {"type": "text", "text": (
            f"The diagram illustrates this question:\n{q_data.get('text', '')}\n\n"
            f"Apply visual checks V1-V4 as instructed."
        )},
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
        print(f"   ⚠️  Diagram rejected (total fails: {fails}): {feedback}...")
        return {
            "diagram_feedback": feedback,
            "total_fail_count": fails,
            "last_failure_type": "diagram",
        }

# ==========================================
# 3. ROUTING
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
def run_seeder(
    exam:             str,
    subject:          str,
    topic:            str,
    sub_topic:        str,
    n_per_level:      int,
    k_iterations:     int,
    difficulty_levels: List[int],
    syllabus_file:    str,
    output_file:      str,
):
    # Build exam-specific prompts once at startup
    sys_prompt  = build_system_prompt(exam)
    crit_prompt = build_critic_prompt(exam)
    exam_cat    = _exam_category(exam)

    print("\n🚀 Starting Question Bank Pipeline...")
    print(f"   Exam        : {exam}  [{exam_cat}]")
    print(f"   Subject     : {subject}")
    print(f"   Topic       : {topic}")
    print(f"   Sub-topic   : {sub_topic}")
    print(f"   N/level     : {n_per_level}  (questions per difficulty per iteration)")
    print(f"   K iterations: {k_iterations}")
    print(f"   Difficulties: {difficulty_levels}")
    print(f"   Haiku       : {_MODEL_HAIKU  or '⚠️  NOT SET — falls back to Sonnet'}")
    print(f"   Sonnet      : {_MODEL_SONNET}")
    print(f"   Opus        : {_MODEL_OPUS}")
    print(f"   Critics     : Factual=Sonnet | Diagram=Sonnet+Vision (visual only)")
    print(f"   Budget      : {MAX_RETRIES} attempts/round, infinite rounds/slot")
    vision_ok = "✅ cairosvg installed" if _CAIROSVG_AVAILABLE else "⚠️  cairosvg missing"
    print(f"   Vision      : {vision_ok}")
    print(f"   Output      : {output_file}")

    # Load syllabus
    with open(syllabus_file, "r") as f:
        full_syllabus = json.load(f)

    # Navigate to the exam's syllabus section
    if exam not in full_syllabus:
        print(f"❌ Exam '{exam}' not found in syllabus. Available: {list(full_syllabus.keys())}")
        return

    exam_data = full_syllabus[exam]

    # Some exams have a sub-key (e.g. UPSC has "General Studies Paper 1 (GS-1)")
    # If top-level has only one key that is a dict of subjects, unwrap it
    if len(exam_data) == 1:
        only_key = list(exam_data.keys())[0]
        if isinstance(exam_data[only_key], dict):
            print(f"   Auto-unwrapping paper: {only_key}")
            exam_data = exam_data[only_key]

    # Resolve target nodes from syllabus
    target_nodes = resolve_target_nodes(exam_data, subject, topic, sub_topic)
    if not target_nodes:
        print(f"❌ No subtopics found for subject='{subject}' topic='{topic}' sub_topic='{sub_topic}'")
        return

    print(f"\n📋 {len(target_nodes)} subtopic(s) to process")

    # Load existing bank
    master_question_bank: List[Dict] = []
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            master_question_bank = json.load(f)
        print(f"   Resuming: {len(master_question_bank)} questions already banked")

    used_numbers: List[str] = [
        fp for q in master_question_bank if (fp := numeric_fingerprint(q))
    ]

    # Question type preference wording — exam-aware
    def _qtype_hint(difficulty: int) -> str:
        if exam_cat == "upsc_gs":
            return "Statement-based or Assertion-Reason" if difficulty >= 3 else "Direct Factual or Matching"
        elif exam_cat == "cloud_cert":
            return "Scenario or Architect trade-off" if difficulty >= 3 else "Conceptual or Best-practice"
        elif exam_cat == "lssbb":
            return "Scenario or Calculation" if difficulty >= 3 else "Definition or Single-step Calculation"
        else:
            return "Application or Scenario" if difficulty >= 3 else "Recall or Direct Application"

    # Main loop: subtopic → difficulty → iteration × n_per_level
    for subj, tpc, st in target_nodes:
        label = f"{subj} → {tpc} → {st}" if tpc else f"{subj} → {st}"
        print(f"\n{'='*58}")
        print(f"🎯  {label}")
        print(f"{'='*58}")

        for difficulty in difficulty_levels:
            for k in range(1, k_iterations + 1):
                for n in range(1, n_per_level + 1):
                    slot_label = f"Level {difficulty} | Iter {k}/{k_iterations} | Q {n}/{n_per_level}"
                    print(f"\n👉  {slot_label}")

                    slug      = re.sub(r'[^A-Z0-9]', '', st.upper())[:10]
                    exam_slug = re.sub(r'[^A-Z0-9]', '', exam.upper())[:6]

                    round_num      = 0
                    total_attempts = 0
                    banked         = False
                    tried_concepts: List[str] = []

                    while not banked:
                        round_num += 1
                        forced_id = f"{exam_slug}_{slug}_{difficulty}_{k}_{n}_{uuid.uuid4().hex[:6]}"

                        extra_hint = ""
                        if round_num > 1:
                            concepts_str = (
                                "\n".join(f"  • {c}" for c in tried_concepts[-5:])
                                if tried_concepts else "  • (none recorded)"
                            )
                            extra_hint = (
                                f"\n- IMPORTANT: {round_num - 1} previous round(s) of "
                                f"{MAX_RETRIES} attempts failed for this slot.\n"
                                f"- Question angles already tried:\n{concepts_str}\n"
                                f"- Choose a completely different angle — different concept, "
                                f"service, or aspect of the sub-topic.\n"
                                f"- Vary question type if the previous type kept failing."
                            )

                        # Build request for this specific slot
                        request_parts = [
                            f"- Exam: {exam}",
                            f"- Subject: {subj}",
                        ]
                        if tpc:
                            request_parts.append(f"- Topic: {tpc}")
                        request_parts += [
                            f"- Sub-topic: {st}",
                            f"- Difficulty Level: {difficulty} / 5",
                            f"- Question type preference: {_qtype_hint(difficulty)}",
                            f"- Diagram_Mode: Auto — reason whether a diagram genuinely aids "
                            f"understanding BEFORE setting Requires_Diagram.",
                        ]
                        request = "\n".join(request_parts) + extra_hint

                        print(f"   🔁 Round {round_num}")

                        initial_state: QuestionState = {
                            "request_prompt":    request,
                            "forced_id":         forced_id,
                            "system_prompt":     sys_prompt,
                            "critic_prompt":     crit_prompt,
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

                        final_state    = app.invoke(initial_state)
                        total_attempts += final_state.get("generation_count", 0)

                        last_q = final_state.get("question_data")
                        if last_q:
                            tried = last_q.get("metadata", {}).get("sub_topic", "")
                            if tried and tried not in tried_concepts:
                                tried_concepts.append(tried)

                        q_data    = final_state.get("question_data")
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

    total = len(master_question_bank)
    print(f"\n✅ Done. {total} questions in {output_file}")


# ==========================================
# ══ RUN CONFIG — EDIT THIS SECTION ════════
# ==========================================
if __name__ == "__main__":

    # ── EXAM SELECTION ─────────────────────────────────────────────────────
    # Exact key from syllabus_maps.json. Options include:
    #   "UPSC CSE Prelims"
    #   "Lean Six Sigma Black Belt (IASSC)"
    #   "SSC CGL"
    #   "AWS Solutions Architect Associate"
    #   ... (see syllabus_maps.json for full list)
    EXAM = "Microsoft Power BI Data Analyst (PL-300)"

    # ── SCOPE SELECTION ────────────────────────────────────────────────────
    # Set any of these to "All" to iterate all options at that level.
    # Examples:
    #   SUBJECT="All", TOPIC="All", SUB_TOPIC="All"  → entire exam
    #   SUBJECT="Indian Polity", TOPIC="All", SUB_TOPIC="All"  → all Polity subtopics
    #   SUBJECT="Indian Polity", TOPIC="All", SUB_TOPIC="Fundamental Rights (Articles 12–35)"
    #   SUBJECT="All", TOPIC="All", SUB_TOPIC="Fundamental Rights (Articles 12–35)"

    SUBJECT   = "Prepare the Data"
    TOPIC     = "Get Data"        # UPSC GS-1 is flat (no topic level), keep as "All"
    SUB_TOPIC = "All"        # "All" = iterate every subtopic under chosen subject

    # ── GENERATION CONFIG ──────────────────────────────────────────────────
    N_PER_LEVEL       = 2          # questions to bank per difficulty level per iteration
    K_ITERATIONS      = 1          # iterations (K=2 doubles the total questions)
    DIFFICULTY_LEVELS = [1,2,3,4,5]

    # ── FILES ──────────────────────────────────────────────────────────────
    SYLLABUS_FILE = "syllabus_maps.json"
    # Set OUTPUT_FILE explicitly per exam to avoid cross-exam collisions.
    OUTPUT_FILE   = "_stats_question_bank.json"
    OUTPUT_FILE = EXAM+"_"+SUBJECT+"_"+TOPIC+"_"+SUB_TOPIC+".json"

    run_seeder(
        exam              = EXAM,
        subject           = SUBJECT,
        topic             = TOPIC,
        sub_topic         = SUB_TOPIC,
        n_per_level       = N_PER_LEVEL,
        k_iterations      = K_ITERATIONS,
        difficulty_levels = DIFFICULTY_LEVELS,
        syllabus_file     = SYLLABUS_FILE,
        output_file       = OUTPUT_FILE,
    )