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
import time
from typing import TypedDict, Optional, Dict, Any, List, Tuple
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from botocore.config import Config
from botocore.exceptions import ClientError

load_dotenv()

# ==========================================
# 0. CONFIG
# ==========================================
MAX_RETRIES       = 10
PIVOT_AFTER_FAILS = 3
RENDERER_URL      = os.getenv("RENDERER_URL", "http://localhost:3002/api/render")

# ── Diagram refinement budget (independent of the text-question budget) ──────
# The visual loop keeps refining a figure until it is clean rather than bailing
# to text-only early. We only pivot away from a diagram after this many failed
# visual refinements, and we never pivot before MIN_DIAGRAM_REFINES attempts.
DIAGRAM_MAX_REFINES   = 6     # hard cap on visual refinement attempts per figure
DIAGRAM_PIVOT_AFTER   = 6     # only suggest text-only fallback after this many visual fails
DIAGRAM_PASS_SCORE    = 4     # 1-5 rubric: a figure must score >= this to be banked
RENDER_DPI            = 170   # rasterisation DPI for the vision critic
MAX_OVERFULL_BOXES    = 0     # reject layouts with more overfull hboxes than this

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
FORMATTING — MANDATORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES:
  **Article 21**, **Article 32** — use standard Markdown bold for articles
  *Kesavananda Bharati v. State of Kerala (1973)* — use Markdown italics for case citations
  **Statement 1:** ... in text field
  **Statement 1 — TRUE:** / **Statement 1 — FALSE:** in explanation
  **Assertion (A):** and  **Reason (R):**
  **1 and 2 only**,  **1, 2 and 3** — combo options

WORKED EXAMPLE — Statement-based:
  "text": "...**Statement 1:** **Article 14** guarantees equality...\\nWhich is/are correct?",
  "options": {{"A": "**1 and 2 only**", "B": "**2 and 3 only**", "C": "**1 only**", "D": "**1, 2 and 3**"}},
  "explanation": "**Statement 1 — TRUE:** **Article 14** uses 'any person'...\\n\\n**Statement 2 — FALSE:** ..."

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
  • Service names in **bold Markdown** — e.g. **Amazon S3**, **AWS IAM**, **Amazon RDS**.

Difficulty scale (SAA-C03 Associate level):
  1 = Identify the correct service for a single stated requirement
  2 = Choose between 2 similar services given a specific constraint
  3 = Multi-constraint scenario (cost + performance, or security + availability)
  4 = Architect-level trade-off analysis across 2-3 services/features
  5 = Complex multi-service solution design with operational and cost optimisation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATTING — MANDATORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES:
  **Amazon S3**, **AWS IAM**, **Amazon RDS** — all AWS service names in bold Markdown
  **IAM Role**, **Security Group**, **VPC** — key technical terms in bold Markdown
  *least privilege*, *shared responsibility model* — principle names in italics Markdown
  **MOST cost-effective**, **LEAST operational overhead** — emphasis words in options

WORKED EXAMPLE:
  "text": "A company stores customer data in **Amazon S3**. The security team requires that all data be encrypted at rest and that encryption keys be rotated annually without application changes. Which solution meets these requirements with the **LEAST operational overhead**?",
  "options": {{
    "A": "Use **SSE-S3** (*Server-Side Encryption with S3-managed keys*)",
    "B": "Use **SSE-KMS** with an **AWS managed key** and enable automatic key rotation",
    "C": "Use **SSE-C** with customer-provided keys rotated by a Lambda function",
    "D": "Use client-side encryption with a custom key management solution"
  }},
  "explanation": "**B is correct:** **SSE-KMS** with an AWS managed key supports automatic annual rotation natively — no application changes required...\\n\\n**A is incorrect:** **SSE-S3** encrypts data but S3-managed keys cannot be configured for custom rotation schedules..."
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
FORMATTING — MANDATORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  **Cpk**, **DPMO**, **RTY** — metric names in Markdown bold
  $C_{{pk}} = \\min\\left[\\frac{{USL-\\mu}}{{3\\sigma}}, \\frac{{\\mu-LSL}}{{3\\sigma}}\\right]$  — formulas inline using $...$
  **Measure Phase**, **Analyse Phase** — DMAIC phases in bold
  **Statement 1 — TRUE:** / **Statement 1 — FALSE:** — in explanations
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

VARIETY WITHIN A SUB-TOPIC (IMPORTANT):
A real paper tests a sub-topic from MANY angles, not the same template repeatedly.
For example, a "Simplification" sub-topic includes: pure surd/indices simplification,
BODMAS order-of-operations, approximation (≈), find-the-missing-term, fraction/decimal
comparison, "what should replace ?", percentage-of-a-value, and word-framed one-liners.
Each new question you generate should use a DIFFERENT structure or phrasing from the
typical one — vary the operations, the question stem wording, and the solving approach.
Never produce the same skeleton (e.g. √a + √b − √c + n²) again and again with new numbers.

Difficulty scale:
  1 = Direct recall or single-step  2 = Two-step or concept distinction
  3 = Multi-step or application     4 = Complex application or reasoning
  5 = Hardest exam-level question

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATTING — MANDATORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  $...$ for inline math, $$...$$ for display math.
  **key term** for important concepts and terms in Markdown bold.
  **Statement N — TRUE/FALSE:** structure in explanations.
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
FORMATTING — MANDATORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  **key term** for important concepts, tools, and service names (Markdown).
  *principle or methodology name* for named principles (Markdown).
  $...$ for any mathematical notation.
  **Statement N — TRUE/FALSE:** in explanation headers.
"""

    # ── Shared tail: JSON escaping, diagram rules + output format ────
    shared_tail = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JSON ESCAPING RULES (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Because you are outputting a JSON string, all backslashes used in LaTeX math MUST be double-escaped.
Example: use $\\\\frac{1}{2}$ instead of $\\frac{1}{2}$.
Example: use \\\\mu instead of \\mu.
Failure to double-escape backslashes will corrupt the JSON parsing.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIAGRAM DECISION — MANDATORY REASONING STEP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You decide whether THIS question needs a figure. Use your real knowledge of how THIS specific
exam's paper actually looks — different exams have very different figure conventions:
  • Some exams put figures on geometry, mensuration, and figure-based reasoning.
  • Bank/PO-style aptitude papers use figures almost ONLY for Data Interpretation
    (bar / line / pie / caselet charts); their arithmetic, algebra, number-series,
    simplification, and word problems are TEXT-ONLY and need no figure.
  • Knowledge/awareness/English/conceptual questions are essentially never figure-based.

Decision test: "Would a real candidate sitting THIS exam normally see a figure with a question
of this exact type? Would the student be UNABLE to answer without it?"

Set "Requires_Diagram": true ONLY when a visual is genuinely part of how this exam asks this
question type (e.g. a DI chart the student must read, a geometry diagram, a process flow a
question literally refers to). Otherwise set it false. When genuinely in doubt → false.

Do NOT add a figure merely because one *could* decorate the question. A figure that just
restates text the student could read anyway is WRONG — omit it.

When a diagram IS needed:
  • Setting Requires_Diagram=true is a COMMITMENT: you MUST then provide valid TikZ_Code that
    renders that figure (and diagram_data if it is a chart). The pipeline will not accept a
    "true" with no drawing, and will not silently convert it to text.
  • TikZ only. Keep it simple, balanced, and uncluttered — labels, boxes, arrows.
  • ANTI-CHEATING: never draw the answer or any computed value in the question diagram.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA-FIRST RULE FOR CHART / GRAPH QUESTIONS (bar, line, pie, scatter, tabular DI)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For ANY data-interpretation chart, the chart IS the data source the student must read.
Follow this order strictly:

  1. DECIDE the underlying data FIRST and put it in the "diagram_data" field (schema below).
     This single object is the ONE source of truth.
  2. The TikZ_Code MUST render EXACTLY those numbers — every bar height, point, or slice
     comes from diagram_data. Do not invent different values in the drawing.
  3. The "explanation" MUST compute using ONLY those same numbers, in ONE pass.
     NEVER show two different datasets, never write "wait, let me recompute with other values",
     never reverse-engineer numbers to hit an option. If your first dataset doesn't yield a
     clean option, CHANGE diagram_data up front and redraw — don't patch mid-explanation.
  4. ANTI-REDUNDANCY: the "text" (question stem) MUST NOT restate the full data series.
     The student reads the values FROM the chart. Phrases like
       "Store P: Jan=80, Feb=95, ..." or "P received 480, Q received 360, ..."
     in the stem are FORBIDDEN for chart questions — that makes the chart pointless.
     The stem may mention at most one anchor value if pedagogically needed, plus any
     EXTRA data not shown on the chart (e.g. "revenue per unit is ₹32,000").
  5. Verify before output: totals/ratios in the explanation must reconcile to the
     numbers in diagram_data exactly. If they don't, fix diagram_data and recompute.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TikZ CODE RULES (when diagram is needed) — FOLLOW EXACTLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PREAMBLE
  1. \\documentclass[border=4mm]{standalone}   ← border-crop, NOT varwidth (avoids huge empty canvas).
  2. \\usepackage{tikz} + EXPLICITLY \\usetikzlibrary every library you use
     (e.g. arrows.meta, positioning, shapes.geometric, fit, backgrounds, calc).

LAYOUT (this is what makes figures clean and human-readable)
  3. Keep the WHOLE picture compact: raw coordinates strictly within -10..+10 on x and -8..+8 on y.
     Do NOT spread nodes across a wide canvas — tight spacing reads better.
  4. fit BOXES MUST be declared AFTER every node they contain. A \\node[fit=...] only
     wraps nodes that already exist — never reference a node created later.
  5. fill=white on any label/node that sits on top of a line or arrow, so it stays legible.
  6. Place edge labels with [midway, sloped] or a white-filled node so they never sit on the line.
  7. Leave clear gaps between boxes (≥0.6cm). Never let two boxes or a box and a border touch/overlap.
  8. No global \\scale transforms. Use consistent node styles defined once in \\begin{tikzpicture}[...].
  9. A legend, if any, goes inside the bounding box of the figure — never floating far to the side.

ARCHETYPE SKELETONS — adapt the closest one, keep it minimal:

  • FLOWCHART / PROCESS (top→down):
    \\begin{tikzpicture}[node distance=1.1cm and 1.4cm,
      box/.style={draw,rounded corners,minimum width=2.6cm,minimum height=0.8cm,align=center,font=\\small}]
      \\node[box] (a) {Start};
      \\node[box,below=of a] (b) {Step};
      \\node[box,below=of b] (c) {End};
      \\draw[-{Stealth}] (a)--(b);  \\draw[-{Stealth}] (b)--(c);
    \\end{tikzpicture}

  • ARCHITECTURE / TOPOLOGY (grouped regions):
    define box/style once, place nodes with positioning, then on a background layer
    wrap each group with \\node[draw,dashed,rounded corners,fit=(n1)(n2)(n3),label=above:Region]{};
    (declare the fit node AFTER n1,n2,n3 — rule 4).

  • BAR CHART (data interpretation):
    draw axes with \\draw[->]; \\foreach for y-ticks; one \\fill rectangle per bar;
    label each bar BELOW the axis. Do NOT print the bar's numeric value on the bar
    (anti-cheating) unless the value is given data, not the answer.

  • GEOMETRY:
    draw the shape with \\draw; label vertices with nodes just OUTSIDE the shape;
    mark given lengths/angles only — never the unknown the question asks for.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a raw JSON object — no markdown fences, no preamble, no text after closing brace.
Use Markdown for text styling (**bold**, *italic*).
Use $...$ for Math, ensuring backslashes are double-escaped.

{
  "id": "PLACEHOLDER",
  "text": "Question text using **bold** and *italic* Markdown. Math uses $\\\\frac{a}{b}$.",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "correct_answer": "A",
  "explanation": "Structured explanation with **key term** formatting.",
  "Requires_Diagram": false,
  "TikZ_Code": null,
  "diagram_data": null,
  "computation": null,
  "metadata": {
    "exam": "",
    "subject": "",
    "topic": "",
    "sub_topic": "",
    "difficulty_level": 1
  }
}

"diagram_data" RULES:
  • null when there is no diagram, OR for non-chart diagrams (architecture, flowchart,
    topology, geometry) where there is no tabular dataset.
  • For CHART/GRAPH questions (bar, line, pie, scatter, tabular DI) it is REQUIRED and is the
    single source of truth the TikZ and explanation must both match exactly:
      {
        "chart_type": "bar" | "line" | "pie" | "scatter" | "table",
        "x_labels": ["P", "Q", "R", "S", "T"],            // categories / x-axis (omit for pie)
        "series": [
          {"name": "2022", "values": [480, 360, 540, 420, 300]},
          {"name": "2023", "values": [560, 450, 480, 500, 390]}
        ],
        "unit": "applications",
        "y_axis_label": "Applications"
      }
    Each series' values align positionally with x_labels. The TikZ bars/points MUST equal
    these values; the explanation MUST compute only from these values.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NUMERIC QUESTIONS — "computation" FIELD (MANDATORY for any calculation question)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If the answer is obtained by ARITHMETIC (simplification, approximation, ratios, percentages,
averages, interest, speed/time, DI calculations, etc.), you MUST emit a "computation" object.
A Python engine evaluates "expression", sets the answer/options authoritatively, and renders
the math in the stem FOR you — so you only need to get the EXPRESSION right.

  1. Write the exact arithmetic as a SINGLE plain expression (no prose) in "expression".
     Use: + - * / , ** or ^ for powers, sqrt(x), cbrt(x), root(x,n), parentheses.
     Percent is allowed as "15% of 1200" or "15%". Example:
       "sqrt(289) - cbrt(125) + 4^3/8"
  2. In the question "text", put the literal token [[EXPR]] exactly where the math should
     appear. Python replaces [[EXPR]] with the properly typeset expression, so the visible
     question is GUARANTEED to match the computation. Do NOT also write the math out by hand.
       e.g. "text": "Simplify the following expression: [[EXPR]]"
            "text": "What is the value of [[EXPR]] ?"
  3. Provide your best "computed_value", four "options", and "correct_answer" — but these are
     finalized by Python, so do not agonise over them (if your options don't contain the true
     value, Python rebuilds them).
  4. For approximation questions, set "tolerance" to the allowed absolute gap (e.g. 0.5).

  "computation": {
    "expression": "(22*3 + 100 - 19)/7",
    "computed_value": 21,
    "tolerance": 0
  }

RULES:
  • PREFER the [[EXPR]] placeholder for any question whose stem is essentially "evaluate this
    expression" (simplification, approximation, BODMAS). It removes all risk of the stem and the
    computation disagreeing.
  • For WORD PROBLEMS (the math is embedded in prose, e.g. speed/time, profit/loss), you may not
    be able to use [[EXPR]] — in that case write the stem normally, but the numbers you state in
    the prose MUST match "expression" exactly, or the question is rejected.
  • DO NOT mention this process anywhere. The student-facing fields ("text", "options",
    "explanation") must NEVER reference Python, an engine, verification, "the answer will be set",
    "[[EXPR]]" leftover, or any tooling. Write them as a finished, published exam question.
  • The "explanation" must be a single clean step-by-step solution. NEVER include "wait",
    "hmm", "let me recheck", alternative attempts, or meta-commentary. If your first numbers
    don't give a clean result, CHANGE the numbers up front and redo — do not narrate it.
  • Set "computation": null ONLY for purely non-numeric questions (conceptual, verbal, GA, English).
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
STEP 4 — TEXT FORMAT CHECK: **Article XX** in text; **Statement N — TRUE/FALSE:** in explanation; **combo** in options. Ensure Markdown bolding is used, NOT LaTeX \\textbf{{}}.
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
STEP 4 — TEXT FORMAT CHECK: **Service names** in text using Markdown bold; key emphasis words bold in options. Ensure it is NOT using LaTeX \\textbf{{}}.
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
STEP 4 — FORMAT CHECK: formulas use $...$ notation; key terms **bold**. Ensure Markdown bolding is used, NOT LaTeX \\textbf{{}}.
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
STEP 4 — FORMAT CHECK: key terms use **bold Markdown** (not \\textbf{{}}); formulas use $...$. 
STEP 5 — ANTI-CHEATING: question stem doesn't reveal the answer."""

    shared_response = """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHART / DATA-INTERPRETATION QUESTIONS (when "diagram_data" is provided)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The "diagram_data" object IS the chart the student reads — treat it as the source of truth.
You are NOT missing a diagram; the numbers are right there in diagram_data.
  • Recompute the answer using ONLY the values in diagram_data. Does it match correct_answer
    and exactly one option? State the value you get.
  • The explanation must use those SAME values in ONE consistent pass. REJECT if the explanation
    shows more than one dataset, recomputes with different numbers, or reverse-engineers values.
  • ANTI-REDUNDANCY: the question "text" must NOT restate the full data series (e.g.
    "P=480, Q=360, R=540 ..."). Reading values off the chart is the skill being tested.
    If the stem duplicates the whole dataset, REJECT and say the data belongs only in the chart.
  • Any EXTRA data needed but not shown on the chart (e.g. price per unit) MAY be in the stem.

RESPONSE:
  Everything correct → reply with ONLY the single word: PASS
  Any issue → short numbered list. State the correct fact/value for every error found.
  For statement-based errors: state which statements are actually T/F and why.
  For chart questions: state the value you computed from diagram_data and the option it matches.
"""
    return domain_block + shared_response

# ==========================================
# 0c. DIAGRAM CRITIC PROMPT (Sonnet + Vision)
# ==========================================
DIAGRAM_CRITIC_PROMPT = """You are a meticulous diagram visual QA reviewer. You receive (1) the rendered
diagram image and (2) the TikZ source that produced it. Your job is purely about whether a
HUMAN STUDENT would find the figure clean, correct, and easy to interpret. You do NOT check
mathematical answers (that is done separately) — but you DO check that the figure faithfully
depicts what the question describes.

Score the figure on each rubric dimension from 1 (broken) to 5 (excellent):

D1. FIDELITY: Does the figure depict exactly what the question states? Count entities — if the
    question says "six VPCs" or "five stores", are exactly that many drawn and labelled correctly?
    Wrong/missing/extra elements relative to the question text → low score.

D2. LAYOUT & CLIPPING: Is everything inside the frame with comfortable margins? Nothing cut off,
    nothing spilling outside a container/region border, no giant empty area dominating the canvas.

D3. LEGIBILITY: Are all labels readable — not overlapping lines, arrows, or each other?
    Edge labels sit clear of the lines. No text collisions.

D4. CLARITY: Would a student immediately understand the structure? Logical flow, consistent
    styling, sensible spacing. Not cluttered, not ambiguous.

D5. ANTI-CHEATING: The diagram must NOT reveal the answer or any computed result.
    If it does, this is an automatic fail regardless of other scores.

RESPONSE FORMAT (exactly):
  Line 1: "SCORE: <min of the five dimensions>" (an integer 1-5)
  Then, if any dimension scored below 4, a short numbered list naming the dimension and the
  SPECIFIC, ACTIONABLE fix (e.g. "D2: the 'Prod RT' box at (-3,-0.8) sits outside the dashed
  region border — move it inside the fit area or extend the region box").
  If D5 fails, set SCORE: 1 and say so explicitly.
  Be concrete and reference coordinates/node names from the source. Do not compute any mathematics.
"""

# ==========================================
# 0d. TIKZ RENDER CLIENT  (server-side SVG + PNG + log warnings)
# ==========================================
def render_tikz(code: str, dpi: int = RENDER_DPI) -> Dict[str, Any]:
    """
    Calls the renderer in JSON mode and returns:
      {"ok": bool, "svg": str|None, "png_base64": str|None,
       "overfull": int, "underfull": int, "warnings": [str], "error": str|None}
    The PNG is produced server-side (pdftoppm) — no cairosvg dependency.
    """
    try:
        res = requests.post(
            RENDERER_URL, json={"code": code, "format": "json", "dpi": dpi}, timeout=120
        )
    except Exception as e:
        return {"ok": False, "svg": None, "png_base64": None, "overfull": 0,
                "underfull": 0, "warnings": [], "error": f"Renderer unreachable: {e}"}

    if res.status_code != 200:
        try:
            err = res.json().get("error", "Unknown error")
        except Exception:
            err = res.text[:500]
        return {"ok": False, "svg": None, "png_base64": None, "overfull": 0,
                "underfull": 0, "warnings": [], "error": err}

    try:
        data = res.json()
    except Exception:
        # Older renderer that returns raw SVG — degrade gracefully.
        return {"ok": True, "svg": res.text, "png_base64": None, "overfull": 0,
                "underfull": 0, "warnings": [], "error": None}

    return {
        "ok": True,
        "svg": data.get("svg"),
        "png_base64": data.get("png_base64"),
        "overfull": int(data.get("overfull", 0) or 0),
        "underfull": int(data.get("underfull", 0) or 0),
        "warnings": data.get("warnings", []) or [],
        "error": None,
    }


# ==========================================
# 0e. DETERMINISTIC TIKZ LINTER
# Cheap structural checks that catch common breakage WITHOUT a render round-trip
# or a vision call. Returns a list of problem strings (empty == clean).
# ==========================================
_TIKZ_LIBRARY_HINTS = {
    "fit":              r"\bfit\s*=",
    "positioning":      r"\b(above|below|left|right)\s*=\s*of\b",
    "arrows.meta":      r"\{?Stealth|Latex|Triangle",
    "backgrounds":      r"on background layer",
    "calc":             r"\$\([^)]*\)\s*[-+!]",
    "shapes.geometric": r"\b(regular polygon|trapezium|ellipse|diamond|cylinder|star)\b",
}

def lint_tikz(code: str) -> List[str]:
    problems: List[str] = []
    if not code or not code.strip():
        return ["empty TikZ code"]

    # Brace balance
    if code.count("{") != code.count("}"):
        problems.append(
            f"unbalanced braces: {code.count('{')} '{{' vs {code.count('}')} '}}'"
        )

    # Document scaffold
    if "\\begin{document}" not in code or "\\end{document}" not in code:
        problems.append("missing \\begin{document}/\\end{document}")
    if "\\begin{tikzpicture}" not in code or "\\end{tikzpicture}" not in code:
        problems.append("missing \\begin{tikzpicture}/\\end{tikzpicture}")

    # Library usage without \usetikzlibrary
    loaded = set(re.findall(r"\\usetikzlibrary\{([^}]*)\}", code))
    loaded_libs = {lib.strip() for group in loaded for lib in group.split(",")}
    body = code
    for lib, pattern in _TIKZ_LIBRARY_HINTS.items():
        if lib not in loaded_libs and re.search(pattern, body):
            problems.append(f"uses '{lib}' features but \\usetikzlibrary{{{lib}}} not loaded")

    # Coordinate bounds (raw explicit coordinates only)
    for m in re.finditer(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", code):
        x, y = float(m.group(1)), float(m.group(2))
        if abs(x) > 14 or abs(y) > 12:
            problems.append(
                f"coordinate ({x},{y}) is far outside the recommended canvas — risks clipping/whitespace"
            )
            break  # one report is enough

    # fit must reference nodes that already exist BEFORE the fit declaration.
    # Position-based (not line-based) so single-line TikZ is handled correctly.
    # A node name is the (name) that follows \node and its optional [..] block,
    # so we skip the fit targets that live inside the options.
    declared_at: Dict[str, int] = {}
    for m in re.finditer(r"\\node\s*(?:\[[^\]]*\])?\s*\(([^)]+)\)", code):
        name = m.group(1).strip()
        # record the EARLIEST declaration position for each node name
        declared_at.setdefault(name, m.start())

    reported: set = set()
    for fm in re.finditer(r"fit\s*=\s*((?:\([^)]*\)\s*)+)", code):
        fit_pos = fm.start()
        for r in re.findall(r"\(([^)]+)\)", fm.group(1)):
            r = r.strip()
            if not r or r in reported:
                continue
            decl = declared_at.get(r)
            if decl is None or decl > fit_pos:
                reported.add(r)
                problems.append(
                    f"fit references node '({r})' that is not declared earlier — "
                    f"declare the fit box AFTER all nodes it wraps"
                )

    return problems


def parse_diagram_score(feedback: str) -> Tuple[int, str]:
    """
    Parses the rubric critic's response. Returns (score 1-5, cleaned_feedback).
    If no SCORE line is found, treats a leading PASS as 5 and otherwise as 2.
    """
    m = re.search(r"SCORE\s*[:=]\s*([1-5])", feedback, re.IGNORECASE)
    if m:
        return int(m.group(1)), feedback.strip()
    if is_pass(feedback):
        return 5, feedback.strip()
    return 2, feedback.strip()

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
    diagram_refines:   int          # visual refinement attempts on the current figure
    diagram_score:     int          # last rubric score (1-5)
    final_image_path:  Optional[str]
    png_b64:           Optional[str] # server-rendered PNG of the current figure
    used_numbers:      List[str]
    image_dir:         str
    diagram_mode:      str          # "auto" | "force" | "never" — manual override

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


def structure_fingerprint(q_data: dict) -> str:
    """
    A number-agnostic signature of a question's STRUCTURE, so we can detect
    "same template, different numbers" repetition (e.g. √a+√b−√c+n² over and over).
    Combines the operations/functions used with a coarse question-type tag.
    """
    text = (q_data.get("text") or "")
    comp = q_data.get("computation") or {}
    expr = comp.get("expression") or ""
    haystack = f"{text} {expr}".lower()

    feats = []
    # Operation / function features (presence, not count).
    checks = [
        ("sqrt",  r"\\sqrt(?!\[)|\bsqrt\("),
        ("cbrt",  r"\\sqrt\[3\]|\bcbrt\(|sqrt\[3\]"),
        ("nroot", r"\\sqrt\[(?!3\])\d|root\("),
        ("pow",   r"\^|\*\*"),
        ("frac",  r"\\d?frac|/"),
        ("pct",   r"%|percent"),
        ("mult",  r"\\times|\*|×"),
        ("ratio", r"\bratio\b|\d\s*:\s*\d"),
        ("avg",   r"average|mean"),
        ("interest", r"interest|principal|per annum|p\.a\."),
        ("speed", r"speed|km/h|distance|train|boat"),
        ("profit", r"profit|loss|cost price|selling price|discount"),
    ]
    for name, pat in checks:
        if re.search(pat, haystack):
            feats.append(name)

    # Coarse question-type tag from the stem phrasing.
    if re.search(r"simplif|value of|evaluate|solve", haystack):
        qtype = "evaluate"
    elif re.search(r"which|find the|how many|what is the (number|total)", haystack):
        qtype = "find"
    elif "?" in text:
        qtype = "wordq"
    else:
        qtype = "other"
    feats.append(f"type:{qtype}")
    return "|".join(feats) if feats else "plain"


def is_pass(feedback: str) -> bool:
    words = feedback.strip().split()
    return bool(words) and words[0].upper().rstrip(".,!:*#") == "PASS"


def needs_diagram(q_data: Optional[dict]) -> bool:
    return bool(q_data and q_data.get("Requires_Diagram") and q_data.get("TikZ_Code"))


# Sub-topics / keywords that mark a question as a data-interpretation CHART, for which
# diagram_data is mandatory and the stem must not restate the whole dataset.
_CHART_KEYWORDS = (
    "bar graph", "bar chart", "line graph", "line chart", "pie chart", "histogram",
    "scatter", "data interpretation", " di", "tabular di", "caselet",
)

def is_chart_question(q_data: dict) -> bool:
    """True if this looks like a data-interpretation chart question."""
    meta = q_data.get("metadata", {}) or {}
    hay = " ".join(str(meta.get(k, "")) for k in ("sub_topic", "topic")).lower()
    if any(k in hay for k in _CHART_KEYWORDS):
        return True
    dd = q_data.get("diagram_data")
    if isinstance(dd, dict) and dd.get("chart_type") in ("bar", "line", "pie", "scatter", "table"):
        return True
    return False


def validate_diagram_data(q_data: dict) -> List[str]:
    """
    Deterministic data-first checks for chart questions. Returns problem strings.
    Catches: missing diagram_data, malformed series, ragged series vs x_labels,
    and the anti-redundancy violation (stem restates the whole data series).
    """
    problems: List[str] = []
    if not is_chart_question(q_data):
        return problems

    dd = q_data.get("diagram_data")
    if not isinstance(dd, dict):
        problems.append(
            "chart question is missing 'diagram_data' — emit the underlying values as a "
            "diagram_data object first (chart_type, x_labels, series[].values), then draw from it"
        )
        return problems

    chart_type = dd.get("chart_type")
    series = dd.get("series")
    x_labels = dd.get("x_labels")

    if chart_type not in ("bar", "line", "pie", "scatter", "table"):
        problems.append(f"diagram_data.chart_type '{chart_type}' is invalid")

    if not isinstance(series, list) or not series:
        problems.append("diagram_data.series must be a non-empty list of {name, values}")
        return problems

    all_values: List[float] = []
    for i, s in enumerate(series):
        vals = s.get("values") if isinstance(s, dict) else None
        if not isinstance(vals, list) or not vals:
            problems.append(f"diagram_data.series[{i}] has no numeric 'values' list")
            continue
        if not all(isinstance(v, (int, float)) for v in vals):
            problems.append(f"diagram_data.series[{i}].values must be all numbers")
        # x_labels must align with each series (except pie, which uses x_labels as slice names)
        if chart_type != "pie" and isinstance(x_labels, list) and len(vals) != len(x_labels):
            problems.append(
                f"diagram_data.series[{i}] has {len(vals)} values but there are "
                f"{len(x_labels)} x_labels — they must align positionally"
            )
        all_values.extend(v for v in vals if isinstance(v, (int, float)))

    # Anti-redundancy: if the stem contains most of the plotted values verbatim, the chart
    # is pointless. Flag when the question text restates a large share of the data series.
    text = q_data.get("text", "")
    text_nums = set(re.findall(r"\d+(?:\.\d+)?", text))
    if len(all_values) >= 4:
        plotted = {("%g" % v) for v in all_values}
        overlap = sum(1 for p in plotted if p in text_nums)
        if overlap >= max(4, int(0.6 * len(plotted))):
            problems.append(
                f"ANTI-REDUNDANCY: the question text restates {overlap}/{len(plotted)} of the "
                f"plotted chart values — for a data-interpretation question the stem must NOT "
                f"reproduce the dataset; the student reads it from the chart. Remove the value "
                f"list from the stem (keep only extra data not shown on the chart)."
            )

    return problems


# ==========================================
# 0f. DETERMINISTIC MATH VERIFIER (sympy)
# LLMs routinely write a numeric expression in the stem, pick "clean" options, and
# never actually evaluate their own expression — producing self-contradictory
# questions (expression says 17.35, options are 18/19/21/22). This verifier
# evaluates the model's declared computation in Python and confirms it equals
# exactly one option == correct_answer, BEFORE any critic call is spent.
# ==========================================
try:
    import sympy as _sympy
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations, implicit_multiplication_application,
        convert_xor,
    )
    _SYMPY_AVAILABLE = True
    _SYMPY_TRANSFORMS = (standard_transformations
                         + (implicit_multiplication_application, convert_xor))
except Exception:
    _SYMPY_AVAILABLE = False


_ALLOWED_EXPR_WORDS = {"sqrt", "cbrt", "root", "Abs", "abs", "pi", "of", "frac", "dfrac"}

def _safe_eval_expr(expr_str: str) -> Optional[float]:
    """
    Evaluate a plain arithmetic expression to a float using sympy.
    Supports sqrt, cbrt, root, **/^ powers, fractions, and implicit multiplication.

    IMPORTANT: returns None whenever the input cannot be parsed RELIABLY (leftover
    LaTeX, unknown symbols, unbalanced braces). It must NEVER silently produce a
    wrong number (e.g. evaluating un-converted LaTeX to 0) — a wrong value would
    cause false STEM↔COMPUTATION mismatches and waste retries.
    """
    if not _SYMPY_AVAILABLE or not isinstance(expr_str, str) or not expr_str.strip():
        return None
    s = expr_str.strip()
    # Normalise common notations the model emits.
    s = s.replace("÷", "/").replace("×", "*").replace("−", "-").replace("·", "*")
    s = s.replace("\\times", "*").replace("\\div", "/").replace("\\cdot", "*")
    s = s.replace("\\left", "").replace("\\right", "")

    # Convert LaTeX radicals/fractions INNERMOST-FIRST in a loop. The [^{}] patterns
    # only match commands whose braces contain no further braces, so repeating the
    # pass peels nesting from the inside out (handles \frac{..}{\sqrt{225}-2} etc.).
    for _ in range(12):
        before = s
        s = re.sub(r"\\sqrt\[\s*(\d+)\s*\]\{([^{}]*)\}", r"root((\2),(\1))", s)  # \sqrt[n]{x}
        s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt((\1))", s)                       # \sqrt{x}
        s = re.sub(r"\\(?:d|t)?frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)  # \frac/\dfrac/\tfrac
        if s == before:
            break

    # Unicode radicals over a parenthesised/number argument.
    s = re.sub(r"∛\s*\(?([\d.]+)\)?", r"cbrt(\1)", s)
    s = re.sub(r"∜\s*\(?([\d.]+)\)?", r"root(\1,4)", s)
    s = re.sub(r"√\s*\(?([\d.]+)\)?", r"sqrt(\1)", s)

    # Percent: "15% of 1200" -> "(15/100)*(1200)"; bare "15%" -> "(15/100)"
    s = re.sub(r"([\d.]+)\s*%\s*of\s*", r"(\1/100)*", s, flags=re.IGNORECASE)
    s = re.sub(r"([\d.]+)\s*%", r"(\1/100)", s)

    # RELIABILITY GUARD: if any LaTeX command, backslash, or brace survived the
    # conversion, we could not parse it cleanly — refuse rather than guess.
    if "\\" in s or "{" in s or "}" in s:
        return None
    # Any alphabetic token that is not an allowed function name → unknown symbol.
    for word in re.findall(r"[A-Za-z]+", s):
        if word not in _ALLOWED_EXPR_WORDS:
            return None

    local = {
        "sqrt": _sympy.sqrt,
        "cbrt": lambda x: _sympy.root(x, 3),
        "root": lambda x, n: _sympy.root(x, n),
        "Abs": _sympy.Abs, "abs": _sympy.Abs,
        "pi": _sympy.pi,
    }
    try:
        expr = parse_expr(s, local_dict=local, transformations=_SYMPY_TRANSFORMS,
                          evaluate=True)
        val = expr.evalf()
        if val.is_real is False:
            return None
        return float(val)
    except Exception:
        return None


def _option_to_float(opt_text: str) -> Optional[float]:
    """
    Convert an option string to its numeric VALUE. If the option is a structured
    expression (e.g. a fraction like \\frac{2.5789}{2}), evaluate it properly so we
    get its true value (1.29), not the first number we see (2.5789) — otherwise a
    nonsensical option would falsely "match" the answer and survive.
    """
    if opt_text is None:
        return None
    t = str(opt_text).strip()
    t = re.sub(r"\*\*", "", t).replace("$", "").replace("₹", "").replace(",", "").strip()

    # If it looks like a structured expression (fraction / operator), evaluate it.
    if "frac" in t or re.search(r"[+\-*/^]\s*\d|\\sqrt|\\cdot|\\times|\\div", t):
        val = _safe_eval_expr(t)
        if val is not None:
            return val

    # Otherwise take the plain number (strip remaining latex/units).
    cleaned = re.sub(r"\\[a-zA-Z]+|\\text\{[^}]*\}|[{}%]", "", t).strip()
    m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    return float(m.group()) if m else None


def _extract_stem_value(text: str) -> Optional[float]:
    """
    Best-effort: pull the math expression out of the question STEM and evaluate it,
    so we can confirm the visible question matches the (verified) computation field.

    Targets simplification/approximation stems where the expression is written in
    LaTeX between $...$ / $$...$$ / \\(...\\) / \\[...\\]. Returns None when no single
    evaluable expression is present (e.g. word problems) — caller then skips the check.
    """
    if not text:
        return None

    # Collect inline/display math spans.
    spans: List[str] = []
    spans += re.findall(r"\$\$(.+?)\$\$", text, re.DOTALL)
    spans += re.findall(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", text, re.DOTALL)
    spans += re.findall(r"\\\((.+?)\\\)", text, re.DOTALL)
    spans += re.findall(r"\\\[(.+?)\\\]", text, re.DOTALL)

    # Keep only spans that actually look like an arithmetic expression: at least one
    # operator/radical AND not an equation/inequality/variable-laden formula.
    candidates: List[str] = []
    for s in spans:
        s = s.strip()
        if not s or "=" in s or "<" in s or ">" in s:
            continue
        if re.search(r"[a-zA-Z]", re.sub(r"sqrt|cbrt|root|frac|dfrac|times|div|cdot|left|right|of|sqrt", "", s)):
            # leftover letters → variables/words, not a pure numeric expression
            continue
        if not re.search(r"[-+*/×÷]|sqrt|cbrt|root|frac|√|∛|∜|\^|\*\*", s):
            continue
        candidates.append(s)

    if not candidates:
        return None
    # Prefer the longest candidate (most complete expression).
    candidates.sort(key=len, reverse=True)
    return _safe_eval_expr(candidates[0])


def _nice_number(x: float):
    """Return an int when x is effectively integral, else a rounded float — for clean answers."""
    if abs(x - round(x)) < 1e-6:
        return int(round(x))
    return round(x, 4)


# ==========================================
# 0g. EXPRESSION → LaTeX  (order-preserving, NO simplification)
# Renders computation.expression into the exact LaTeX the student sees, so the
# visible stem can never diverge from the verified math. We do NOT use sympy's
# latex() because evaluate=False still reorders terms and won't emit \sqrt.
# ==========================================
_EXPR_TOKEN_RE = re.compile(r"\s*(\*\*|//|[()+\-*/,]|sqrt|cbrt|root|\d+\.\d+|\d+)")

def _expr_tokenize(s: str) -> Optional[List[str]]:
    toks, i = [], 0
    s = s.replace("^", "**")
    while i < len(s):
        if s[i].isspace():
            i += 1; continue
        m = _EXPR_TOKEN_RE.match(s, i)
        if not m or m.start() != i:
            return None  # unknown character → cannot render reliably
        toks.append(m.group(1)); i = m.end()
    return toks


class _ExprParser:
    def __init__(self, toks): self.t = toks; self.i = 0
    def peek(self): return self.t[self.i] if self.i < len(self.t) else None
    def nxt(self):
        tok = self.peek(); self.i += 1; return tok


def _strip_parens(x: str) -> str:
    x = x.strip()
    if x.startswith("(") and x.endswith(")"):
        return x[1:-1].strip()
    return x


def _le_expr(p: "_ExprParser") -> str:
    out = _le_term(p)
    while p.peek() in ("+", "-"):
        op = p.nxt(); out = f"{out} {op} {_le_term(p)}"
    return out


def _le_term(p: "_ExprParser") -> str:
    out = _le_factor(p)
    while p.peek() in ("*", "/"):
        op = p.nxt(); rhs = _le_factor(p)
        if op == "/":
            out = f"\\frac{{{_strip_parens(out)}}}{{{_strip_parens(rhs)}}}"
        else:
            out = f"{_strip_parens(out)} \\times {_strip_parens(rhs)}"
    return out


def _le_factor(p: "_ExprParser") -> str:
    base = _le_unary(p)
    if p.peek() == "**":
        p.nxt(); exp_raw = _le_factor(p)
        e = _strip_parens(exp_raw)
        # Pretty radicals: x**0.5 → √x , x**(1/3) → ∛x , x**(1/n) → nth root
        if e in ("0.5", "1/2", "\\frac{1}{2}"):
            return f"\\sqrt{{{_strip_parens(base)}}}"
        mroot = re.fullmatch(r"\\frac\{1\}\{(\d+)\}", e) or re.fullmatch(r"1/(\d+)", e)
        if mroot:
            n = mroot.group(1)
            return f"\\sqrt[{n}]{{{_strip_parens(base)}}}"
        return f"{base}^{{{e}}}"
    return base


def _le_unary(p: "_ExprParser") -> str:
    if p.peek() == "-":
        p.nxt(); return f"-{_le_base(p)}"
    return _le_base(p)


def _le_base(p: "_ExprParser") -> str:
    tok = p.peek()
    if tok in ("sqrt", "cbrt", "root"):
        p.nxt()
        if p.nxt() != "(":
            raise ValueError("expected (")
        a = _le_expr(p)
        if p.peek() == ",":
            p.nxt(); n = _le_expr(p)
            if p.nxt() != ")":
                raise ValueError("expected )")
            return f"\\sqrt[{_strip_parens(n)}]{{{_strip_parens(a)}}}"
        if p.nxt() != ")":
            raise ValueError("expected )")
        if tok == "sqrt":
            return f"\\sqrt{{{_strip_parens(a)}}}"
        return f"\\sqrt[3]{{{_strip_parens(a)}}}"
    if tok == "(":
        p.nxt(); inner = _le_expr(p)
        if p.nxt() != ")":
            raise ValueError("expected )")
        return f"({inner})"
    if tok is None:
        raise ValueError("unexpected end")
    p.nxt()
    return tok  # number


def expr_to_latex(expr: str) -> Optional[str]:
    """
    Convert a plain computation expression into order-preserving LaTeX, or None if it
    cannot be parsed cleanly. Used to render the visible question stem FROM the verified
    expression so the two can never diverge.
    """
    if not isinstance(expr, str) or not expr.strip():
        return None
    try:
        toks = _expr_tokenize(expr)
        if not toks:
            return None
        p = _ExprParser(toks)
        out = _le_expr(p)
        if p.i != len(toks):
            return None  # trailing junk → not a clean single expression
        return out.strip()
    except Exception:
        return None


# Token the model puts in the stem where the math expression should be rendered.
_STEM_MATH_PLACEHOLDER = "[[EXPR]]"


def _format_like(value, template: Optional[str]) -> str:
    """
    Render a numeric `value`, reusing the template's wrapper ($...$, **bold**, ₹)
    ONLY when the template is a SIMPLE single-number option. If the template is a
    structured expression (fraction, multiple numbers, operators like \\frac{x}{2}),
    reusing it would corrupt the value, so we fall back to a clean numeric option.
    """
    nv = _nice_number(value)
    num = str(nv)
    if not template:
        return num
    t = str(template)

    nums = re.findall(r"-?\d+(?:\.\d+)?", t)
    # "Simple" = exactly one number and no fraction/operator structure in the body.
    body = re.sub(r"\*\*|\$|\\mathbf|\\text\{[^}]*\}|[{}₹,%\s]|\\[a-zA-Z]+", "", t)
    structured = bool(re.search(r"[+\-*/^]", body.replace(num, ""))) or "frac" in t or len(nums) != 1
    if structured:
        # Preserve only a plain $...$ / bold wrapper style, not the broken structure.
        if t.strip().startswith("$"):
            return f"${num}$"
        if t.strip().startswith("**"):
            return f"**{num}**"
        return num
    return re.sub(r"-?\d+(?:\.\d+)?", num, t, count=1)


def _build_options(actual: float, existing: Dict[str, str]) -> Tuple[Dict[str, str], str]:
    """
    Build a 4-option set guaranteed to contain the true value, with plausible
    numeric distractors, styled like the model's existing options. Returns
    (options_dict, correct_key). Distractors are near-value / common-error offsets —
    standard for simplification/approximation/DI questions.
    """
    template = next((v for v in (existing or {}).values() if v), None)
    is_int = abs(actual - round(actual)) < 1e-6
    base = abs(actual) if actual != 0 else 1.0

    # Candidate distractor deltas (relative + small absolute), avoiding collisions.
    if is_int:
        deltas = [2, -2, 4, -4, 6, -6, 1, -1, 10, -10]
    else:
        step = max(round(base * 0.05, 2), 0.1)
        deltas = [step, -step, 2 * step, -2 * step, 3 * step, -3 * step]

    distractors: List[float] = []
    seen = {round(actual, 4)}
    for d in deltas:
        cand = _nice_number(actual + d)
        # keep positive where the true value is positive (avoid implausible negatives)
        if actual > 0 and cand <= 0:
            continue
        if round(float(cand), 4) in seen:
            continue
        seen.add(round(float(cand), 4))
        distractors.append(cand)
        if len(distractors) == 3:
            break

    # Fallback if we couldn't find 3 (e.g. tiny values): widen.
    extra = 1
    while len(distractors) < 3:
        cand = _nice_number(actual + (base * 0.1 * extra))
        if round(float(cand), 4) not in seen:
            seen.add(round(float(cand), 4)); distractors.append(cand)
        extra += 1

    values = [actual] + distractors
    # Sort ascending so the correct answer isn't always in the same position,
    # but vary the position deterministically by the integer part.
    values_sorted = sorted(values)
    keys = ["A", "B", "C", "D"]
    opts = {k: _format_like(v, template) for k, v in zip(keys, values_sorted)}
    correct_key = keys[values_sorted.index(actual)]
    return opts, correct_key


def verify_computation(q_data: dict, apply_fixes: bool = False,
                       fixes_out: Optional[List[str]] = None) -> List[str]:
    """
    Deterministic numeric check on the model's 'computation' object.

    Two modes:
      • apply_fixes=False (default, "verify only"): returns a list of problem strings;
        empty == clean. Reports answer-key/value/option errors as problems.
      • apply_fixes=True ("authoritative"): PYTHON OWNS THE MATH. It evaluates the
        expression and becomes the source of truth for the ANSWER and the OPTIONS:
          - computed_value  → set to Python's result
          - options         → if they already contain the true value (one match), kept;
                              otherwise REBUILT around the true value with plausible
                              distractors (eliminates the "matches NONE / matches MULTIPLE"
                              retry loop entirely)
          - correct_answer  → set to the option equal to the true value
        Only TWO things remain hard rejects (Python must not silently author them):
          - the expression cannot be evaluated at all, and
          - the visible STEM evaluates to a different number than the expression
            (the student would see a different question than the one we answered).

    computation = {
      "expression": "sqrt(289) - cbrt(125) + 4^3/8",   # plain, evaluable
      "computed_value": 17.35,                           # model's result (may be overwritten)
      "tolerance": 0.01                                  # optional, for approximation Qs
    }
    """
    comp = q_data.get("computation")
    if not isinstance(comp, dict):
        return []  # not a verifiable numeric question; nothing to check

    if not _SYMPY_AVAILABLE:
        return []  # cannot verify without sympy; don't block

    expr = comp.get("expression")
    actual = _safe_eval_expr(expr) if expr else None
    tol = comp.get("tolerance")
    try:
        tol = float(tol) if tol is not None else None
    except Exception:
        tol = None

    # ── HARD problem: expression cannot be evaluated ───────────────────────────
    if actual is None:
        if expr:
            return [
                f"computation.expression '{expr}' could not be evaluated in Python — "
                f"rewrite it as a plain arithmetic expression (use sqrt(), cbrt(), root(x,n), "
                f"**, /, parentheses) with no prose."
            ]
        return []

    # Tolerance: default to exact-ish for integer results, looser for approximation Qs.
    if tol is None:
        tol = max(0.05, abs(actual) * 0.02)

    options = q_data.get("options", {}) or {}
    opt_vals = {k: _option_to_float(v) for k, v in options.items()}
    matches = [k for k, v in opt_vals.items() if v is not None and abs(v - actual) <= tol]
    stem_val = _extract_stem_value(q_data.get("text", ""))

    # ── HARD reject: the VISIBLE stem describes a different problem than we solved.
    # Python must never author an answer for a question the student isn't being shown.
    if stem_val is not None and abs(stem_val - actual) > max(tol, 1e-6):
        msg = (
            f"STEM ↔ COMPUTATION mismatch: the question text evaluates to {stem_val:.4f}, but the "
            f"computation field evaluates to {actual:.4f}. The visible question must use the SAME "
            f"numbers as the computation. Rewrite the stem so it matches the verified expression "
            f"'{expr}' exactly (do not change the numbers only in the computation field)."
        )
        # This is hard in BOTH modes.
        problems = [msg]
        # also surface value/key issues in verify-only mode for completeness
        return problems

    claimed = comp.get("computed_value")
    try:
        claimed = float(claimed) if claimed is not None else None
    except Exception:
        claimed = None
    value_wrong = (claimed is not None and abs(claimed - actual) > tol)
    correct = q_data.get("correct_answer")

    # ── AUTHORITATIVE MODE: Python owns the answer AND the options. ───────────
    if apply_fixes:
        if value_wrong or claimed is None:
            comp["computed_value"] = _nice_number(actual)
            if fixes_out is not None and value_wrong:
                fixes_out.append(f"computed_value {claimed} → {_nice_number(actual)} (Python)")
        if len(matches) == 1:
            # Options already contain the true value exactly once — keep them, fix key.
            if correct != matches[0]:
                q_data["correct_answer"] = matches[0]
                if fixes_out is not None:
                    fixes_out.append(
                        f"correct_answer '{correct}' → '{matches[0]}' (matches {actual:.4f})")
        else:
            # No match, or multiple matches → Python rebuilds the option set so the
            # true value is present exactly once with plausible distractors.
            new_opts, new_key = _build_options(actual, options)
            q_data["options"] = new_opts
            q_data["correct_answer"] = new_key
            if fixes_out is not None:
                reason = "no option matched" if not matches else "multiple options matched"
                fixes_out.append(
                    f"options rebuilt by Python ({reason}); answer = {new_key} ({_nice_number(actual)})")
        # Keep the model's redundant computation.answer_option in sync with the
        # authoritative correct_answer (a stale value here triggers a critic reject).
        if "answer_option" in comp:
            comp["answer_option"] = q_data["correct_answer"]
        return []  # math is now fully Python-authoritative

    # ── VERIFY-ONLY MODE: report everything as problems (used by tests). ──────
    problems: List[str] = []
    if not matches:
        problems.append(
            f"the expression '{expr}' evaluates to {actual:.4f}, which matches NONE of the "
            f"options {opt_vals}. Redesign the numbers so the expression yields exactly one option."
        )
    elif len(matches) > 1:
        problems.append(
            f"the expression evaluates to {actual:.4f}, which matches MULTIPLE options {matches} — "
            f"options are too close; make distractors distinct."
        )
    if value_wrong:
        problems.append(
            f"computed_value {claimed} ≠ Python evaluation {actual:.4f} of the expression "
            f"'{expr}'. Your stated result does not match your own expression."
        )
    if matches and correct not in matches:
        problems.append(
            f"the expression evaluates to {actual:.4f} = option {matches[0]}, but correct_answer "
            f"is set to '{correct}'. Set correct_answer to '{matches[0]}'."
        )
    return problems


def regenerate_explanation(q_data: dict) -> None:
    """
    After Python has finalized the answer/options for a numeric question, rewrite the
    explanation from the VERIFIED numbers in one clean pass (no scratchpad, no
    second-guessing, no mention of Python/pipeline). Mutates q_data in place.
    Falls back to leaving the explanation untouched if the LLM call fails.
    """
    comp = q_data.get("computation")
    if not isinstance(comp, dict):
        return
    expr = comp.get("expression")
    value = comp.get("computed_value")
    correct = q_data.get("correct_answer")
    options = q_data.get("options", {})
    if expr is None or value is None:
        return

    sys_msg = (
        "You write clean, professional exam answer explanations. Output ONLY the explanation "
        "text (Markdown, $...$ for math). Rules: present a single, linear, step-by-step "
        "solution that arrives at the given answer. NEVER include second-guessing, "
        "'wait/hmm/let me recheck', alternative attempts, or ANY mention of tools, Python, "
        "verification, or this instruction. Do not restate the full question."
    )
    human = (
        f"Question: {q_data.get('text','')}\n"
        f"The verified expression is: {expr}\n"
        f"It evaluates to: {value}\n"
        f"Correct option: {correct} = {options.get(correct, value)}\n"
        f"Options: {json.dumps(options, ensure_ascii=False)}\n\n"
        f"Write the clean step-by-step explanation that reaches {value} (option {correct}). "
        f"Optionally end with one short line on why the other options are wrong."
    )
    try:
        resp = safe_invoke(_MODEL_SONNET, [SystemMessage(content=sys_msg),
                                           HumanMessage(content=human)], max_tokens=900)
        text = (resp.content or "").strip()
        if text:
            q_data["explanation"] = text
    except Exception as e:
        print(f"   ⚠️  Explanation regen failed ({e}); keeping original.")


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

# Network/transient errors worth waiting out (Wi-Fi reconnect, wake-from-sleep, throttling).
_TRANSIENT_ERR_MARKERS = (
    "could not connect to the endpoint",
    "failed to resolve",
    "name or service not known",
    "nodename nor servname",
    "connection reset",
    "connection aborted",
    "timed out",
    "read timeout",
    "temporarily unavailable",
    "throttl",            # ThrottlingException / Too Many Requests
    "service unavailable",
    "503", "500", "endpointconnectionerror",
)

def _is_transient(err: Exception) -> bool:
    msg = str(err).lower()
    return any(m in msg for m in _TRANSIENT_ERR_MARKERS)


def safe_invoke(model_id: str, messages: list, max_tokens: int = 8192,
                retries: int = 8, base_delay: float = 3.0, max_delay: float = 60.0):
    """
    Safely invokes the LLM with resilient retries.

    A brief network drop (Wi-Fi reconnect, laptop wake, DNS hiccup) or AWS throttling
    must NOT kill a long run. Transient/connection errors are retried with exponential
    backoff for a generous window (~minutes); the client is recreated each attempt to
    refresh the AWS SigV4 timestamp after sleep. Non-transient errors fail fast.
    """
    delay = base_delay
    for attempt in range(retries):
        try:
            # Recreate the client each attempt → fresh SigV4 timestamp after any sleep.
            llm = make_llm(model_id, max_tokens=max_tokens)
            return llm.invoke(messages)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            transient = (error_code in ("InvalidSignatureException", "ThrottlingException",
                                        "ServiceUnavailableException", "ModelTimeoutException")
                         or _is_transient(e))
            if transient and attempt < retries - 1:
                print(f"   ⚠️ AWS error ({error_code or 'connection'}). "
                      f"Retry {attempt+1}/{retries} in {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
                continue
            raise e
        except Exception as e:
            if _is_transient(e) and attempt < retries - 1:
                print(f"   ⚠️ Network/LLM error: {str(e)[:90]}. "
                      f"Retry {attempt+1}/{retries} in {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
                continue
            if attempt < retries - 1:
                # Unknown error — give it a couple of shorter retries before giving up.
                print(f"   ⚠️ LLM error: {str(e)[:90]}. Retry {attempt+1}/{retries}...")
                time.sleep(base_delay)
                continue
            raise e

# ==========================================
# 1b. SYLLABUS RESOLVER
# Converts (exam, subject, topic, sub_topic) with "All" wildcards
# into a flat list of (subject, topic, sub_topic) tuples to iterate over.
# Works with both flat (subject > [subtopics]) and nested (subject > topic > [subtopics]).
# ==========================================
def unwrap_exam_data(full_syllabus: dict, exam: str) -> Optional[dict]:
    """Return the subject-level dict for an exam, applying the single-paper auto-unwrap
    (mirrors run_seeder). Returns None if the exam is absent."""
    if exam not in full_syllabus:
        return None
    exam_data = full_syllabus[exam]
    if isinstance(exam_data, dict) and len(exam_data) == 1:
        only_key = list(exam_data.keys())[0]
        if isinstance(exam_data[only_key], dict):
            exam_data = exam_data[only_key]
    return exam_data


def list_scope(exam_data: dict):
    """
    Introspect an (unwrapped) exam's syllabus and return:
      { "flat": bool,
        "subjects": [...],
        "topics_by_subject": {subject: [topics]},      # empty lists when flat
        "subtopics": {(subject, topic): [subtopics]} }  # topic="" when flat
    Used by the GUI to build cascading selectors.
    """
    subjects, topics_by_subject, subtopics = [], {}, {}
    flat = True
    for subj, val in exam_data.items():
        subjects.append(subj)
        if isinstance(val, list):
            topics_by_subject[subj] = []
            subtopics[(subj, "")] = list(val)
        elif isinstance(val, dict):
            flat = False
            topics_by_subject[subj] = list(val.keys())
            for tp, tval in val.items():
                subtopics[(subj, tp)] = list(tval) if isinstance(tval, list) else [tval]
    return {"flat": flat, "subjects": subjects,
            "topics_by_subject": topics_by_subject, "subtopics": subtopics}


def _matches(value: str, selector) -> bool:
    """
    True if `value` is selected. `selector` may be:
      • "All"           → match everything
      • a single string → exact match
      • a list/tuple/set of strings → match if value is in it ("All" inside also matches all)
    """
    if selector == "All" or selector is None:
        return True
    if isinstance(selector, (list, tuple, set)):
        if "All" in selector:
            return True
        return value in selector
    return value == selector


def resolve_target_nodes(
    syllabus: dict,
    subject="All",
    topic="All",
    sub_topic="All",
) -> List[Tuple[str, str, str]]:
    """
    Returns a list of (subject, topic, sub_topic) tuples.
    topic is set to "" when the syllabus is flat (subject > [subtopics]).

    subject / topic / sub_topic each accept "All", a single name, OR a list of names.
    """
    nodes: List[Tuple[str, str, str]] = []

    for subj_name, subj_val in syllabus.items():
        if not _matches(subj_name, subject):
            continue

        if isinstance(subj_val, list):
            # Flat structure: subject > [subtopics]  (e.g. UPSC GS-1)
            for st in subj_val:
                if _matches(st, sub_topic):
                    nodes.append((subj_name, "", st))

        elif isinstance(subj_val, dict):
            # Nested structure: subject > topic > [subtopics]  (e.g. IASSC, SSC)
            for topic_name, topic_val in subj_val.items():
                if not _matches(topic_name, topic):
                    continue
                subtopics = topic_val if isinstance(topic_val, list) else [topic_val]
                for st in subtopics:
                    if _matches(st, sub_topic):
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

    # Manual override of the diagram decision (set in RUN CONFIG via DIAGRAM_MODE).
    diagram_mode = state.get("diagram_mode", "auto")
    if diagram_mode == "force":
        prompt += (
            "\n\nDIAGRAM OVERRIDE: This question MUST include a figure. Set "
            "\"Requires_Diagram\": true and provide valid TikZ_Code (and diagram_data if it is a "
            "chart). Choose a question angle for which a figure is genuinely meaningful."
        )
    elif diagram_mode == "never":
        prompt += (
            "\n\nDIAGRAM OVERRIDE: This question MUST be text-only. Set "
            "\"Requires_Diagram\": false and \"TikZ_Code\": null. Do not produce any figure; "
            "phrase the question so it is fully solvable from text alone."
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
        refines = state.get("diagram_refines", 0)
        last_score = state.get("diagram_score", 0)
        # OVERRIDE RULE: once a question requires a diagram, we MUST deliver a diagram.
        # We never convert it to text-only. After many failed refinements we simplify
        # the FIGURE (not abandon it).
        if refines >= DIAGRAM_PIVOT_AFTER:
            print(f"   Mode: Diagram simplify (visual refines: {refines})")
            prompt += (
                f"\n\nThe diagram could not be made clean after {refines} refinement attempts. "
                f"Generate a MUCH SIMPLER figure for the SAME question (fewer nodes, wider spacing, "
                f"no legend, plain boxes/axes). The question still REQUIRES a diagram — keep "
                f"Requires_Diagram=true and provide valid TikZ_Code. Do NOT remove the figure or "
                f"convert the question to text-only. Keep the question text/answer the same. Raw JSON only."
            )
        else:
            print(f"   Mode: Refining diagram (visual refine #{refines + 1}, last score {last_score}/5)")
            prompt += (
                f"\n\nPrevious JSON:\n```json\n{prev_json}\n```\n\n"
                f"A visual QA reviewer scored the rendered figure {last_score}/5 and asked for these fixes:\n"
                f"<diagram_feedback>\n{state['diagram_feedback']}\n</diagram_feedback>\n\n"
                f"Keep question text/options/answer/explanation EXACTLY the same. ONLY revise TikZ_Code "
                f"to address every point above. Re-check: fit boxes declared after their nodes, no element "
                f"outside a region border, no label on top of a line, compact canvas. "
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

    # llm = make_llm(model_id, max_tokens=8192)
    # response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=prompt)])
    response = safe_invoke(model_id, [SystemMessage(content=sys_prompt), HumanMessage(content=prompt)], max_tokens=8192)

    raw = extract_json(response.content)
    try:
        q_data = json.loads(raw)
        q_data["id"] = state["forced_id"]

        # MANUAL OVERRIDE — "never": strip any figure the model produced regardless.
        # Enforced deterministically so it holds even if the model ignores the instruction.
        if diagram_mode == "never" and q_data.get("Requires_Diagram"):
            print("   ⛔ DIAGRAM_MODE=never — stripping the model's figure (text-only enforced)")
            q_data["Requires_Diagram"] = False
            q_data["TikZ_Code"] = None
            q_data["diagram_data"] = None

        # OVERRIDE RULE: Requires_Diagram=true MUST come with usable TikZ_Code.
        # If the model asked for a diagram but supplied no drawing, we do NOT bank it
        # as text — we send it back to produce the figure it said the question needs.
        if q_data.get("Requires_Diagram") and not (q_data.get("TikZ_Code") or "").strip():
            print("   🔧 Requires_Diagram=true but TikZ_Code is missing — regenerating with figure")
            return {
                "raw_json_str":      json.dumps(q_data),
                "question_data":     q_data,
                "generation_count":  gen_count + 1,
                "compile_error":     (
                    "This question has Requires_Diagram=true but no TikZ_Code. The diagram is "
                    "mandatory for this question — provide valid TikZ_Code (and diagram_data if it "
                    "is a chart). Do not set Requires_Diagram=false."
                ),
                "math_feedback":     None,
                "diagram_feedback":  None,
                "total_fail_count":  state.get("total_fail_count", 0) + 1,
                "last_failure_type": "compile",
            }

        # STEM-FROM-COMPUTATION: if the model put the [[EXPR]] placeholder in the stem,
        # render the verified computation.expression into LaTeX and substitute it. The
        # visible question then CANNOT diverge from the math we verify (no more
        # STEM↔COMPUTATION mismatch retries for this style of question).
        comp0 = q_data.get("computation")
        if (isinstance(comp0, dict) and _STEM_MATH_PLACEHOLDER in (q_data.get("text") or "")):
            rendered = expr_to_latex(comp0.get("expression", ""))
            if rendered:
                q_data["text"] = q_data["text"].replace(
                    _STEM_MATH_PLACEHOLDER, f"${rendered}$")
                print("   🧩 Stem rendered from verified expression")
            else:
                # Expression unrenderable → drop the placeholder so we don't ship it,
                # and let verification/critic handle the (now plain) stem.
                q_data["text"] = q_data["text"].replace(_STEM_MATH_PLACEHOLDER, "").strip()

        # DETERMINISTIC MATH CHECK — Python is AUTHORITATIVE for the answer.
        # If the question is sound (expression evaluates, matches exactly one option,
        # stem agrees), Python sets computed_value/correct_answer itself instead of
        # bouncing it back to the LLM. Only hard-broken math (no/multiple match, stem
        # drift, unevaluable expression) is rejected for regeneration.
        math_fixes: List[str] = []
        math_problems = verify_computation(q_data, apply_fixes=True, fixes_out=math_fixes)
        if math_problems:
            print(f"   🧮 Math verifier rejected (Python): {math_problems[0][:110]}")
            return {
                "raw_json_str":      json.dumps(q_data),
                "question_data":     q_data,
                "generation_count":  gen_count + 1,
                "math_feedback":     (
                    "Python verified your math and it does not check out:\n"
                    + "\n".join(f"- {p}" for p in math_problems)
                    + "\nRebuild so the expression, options, and stem are all mutually "
                      "consistent in ONE pass. Do not reverse-engineer."
                ),
                "compile_error":     None,
                "diagram_feedback":  None,
                "total_fail_count":  state.get("total_fail_count", 0) + 1,
                "last_failure_type": "math",
            }
        if math_fixes:
            # Question was sound; Python authored the answer/options. The previous
            # explanation now argues for stale numbers, so regenerate it cleanly from
            # the verified result before it reaches the critic.
            print(f"   🧮 Math auto-corrected by Python: {'; '.join(math_fixes)}")
            regenerate_explanation(q_data)
            print("   📝 Explanation regenerated from verified result")

        # Deterministic checks BEFORE we spend a render/vision/critic call.
        if needs_diagram(q_data):
            pre_problems = lint_tikz(q_data["TikZ_Code"]) + validate_diagram_data(q_data)
            if pre_problems:
                print(f"   🔧 Pre-render checks flagged {len(pre_problems)} issue(s) — fixing first")
                return {
                    "raw_json_str":      json.dumps(q_data),
                    "question_data":     q_data,
                    "generation_count":  gen_count + 1,
                    "compile_error":     "Diagram pre-check issues:\n" + "\n".join(f"- {p}" for p in pre_problems),
                    "math_feedback":     None,
                    "diagram_feedback":  None,
                    "total_fail_count":  state.get("total_fail_count", 0) + 1,
                    "last_failure_type": "compile",
                }

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

    # If the generator's lint step already rejected the TikZ, don't waste a render —
    # propagate so routing sends us back to the generator.
    if state.get("compile_error") and state.get("last_failure_type") == "compile":
        print(f"   ⏭️  Skipping render — lint already flagged issues")
        return {}

    # If the deterministic math verifier already rejected it, skip render AND the
    # critic — route straight back to the generator with the Python feedback.
    if state.get("math_feedback") and state.get("last_failure_type") == "math":
        print(f"   ⏭️  Skipping render/critic — math verifier already rejected")
        return {}

    if not q_data or not q_data.get("Requires_Diagram") or not q_data.get("TikZ_Code"):
        return {"compile_error": None, "final_image_path": None, "png_b64": None}

    print("\n🎨 [Compiler] Rendering diagram...")
    result = render_tikz(q_data["TikZ_Code"])

    if not result["ok"]:
        err = result["error"] or "Unknown error"
        print(f"   ❌ Compile error: {str(err)[:120]}")
        return {
            "compile_error":     err,
            "final_image_path":  None,
            "total_fail_count":  state.get("total_fail_count", 0) + 1,
            "last_failure_type": "compile",
        }

    # Deterministic layout gate: too many overfull boxes ⇒ something is clipped/oversized.
    if result["overfull"] > MAX_OVERFULL_BOXES:
        sample = "\n".join(f"  {w}" for w in result["warnings"][:6])
        print(f"   ⚠️  {result['overfull']} overfull box(es) — layout likely clipped")
        return {
            "compile_error":     (
                f"Layout problem: {result['overfull']} overfull box(es) detected — content is "
                f"too wide/tall and is being clipped or pushed off the page. Tighten the layout, "
                f"shorten labels, or reduce spacing.\nSample warnings:\n{sample}"
            ),
            "final_image_path":  None,
            "total_fail_count":  state.get("total_fail_count", 0) + 1,
            "last_failure_type": "compile",
        }

    gen      = state.get("generation_count", 0)
    img_dir  = state.get("image_dir", "local_images")
    img_name = f"{q_data['id']}_a{gen}.svg"
    img_path = os.path.join(img_dir, img_name)
    os.makedirs(img_dir, exist_ok=True)
    with open(img_path, "w", encoding="utf-8") as f:
        f.write(result["svg"] or "")
    print(f"   ✅ Saved {img_name}")
    return {
        "compile_error":    None,
        "final_image_path": img_path,
        "png_b64":          result["png_base64"],
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

    # Strip the verbose TikZ source (the critic verifies facts, not drawing syntax),
    # but KEEP diagram_data — that structured object IS the chart the student reads, and
    # the critic must verify the answer against it instead of declaring "no diagram".
    q_for_critic = {k: v for k, v in q_data.items() if k != "TikZ_Code"}
    note = ""
    if q_data.get("diagram_data"):
        note = (
            "\n\nNOTE: This is a chart/data-interpretation question. The 'diagram_data' field below "
            "contains the EXACT values plotted in the figure — the student reads these off the chart. "
            "Recompute the answer from diagram_data and verify the explanation uses only these values."
        )
    feedback_response = safe_invoke(_MODEL_SONNET, [
        SystemMessage(content=crit_prompt),
        HumanMessage(content=f"Review:\n```json\n{json.dumps(q_for_critic, indent=2)}\n```{note}"),
    ], max_tokens=1024)
    feedback = feedback_response.content.strip()

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
    png_b64  = state.get("png_b64")
    refines  = state.get("diagram_refines", 0)
    print("\n📐 [DiagramCritic/Sonnet+Vision] Visual rubric check...")

    if not needs_diagram(q_data):
        return {"diagram_feedback": None}

    # A missing image is NOT a pass — we could not verify the figure, so treat it
    # as a failure and route back to regenerate. (Previously this silently passed.)
    if not png_b64:
        fails = state.get("total_fail_count", 0) + 1
        print(f"   ❌ No rendered image to review — cannot verify figure (total fails: {fails})")
        return {
            "diagram_feedback": (
                "The figure could not be rendered to an image for visual review. "
                "Regenerate the TikZ so it compiles to a valid image."
            ),
            "diagram_refines":  refines + 1,
            "diagram_score":    0,
            "total_fail_count": fails,
            "last_failure_type": "diagram",
        }

    print("   🖼️  Image + source loaded for visual review")
    data_note = ""
    if q_data.get("diagram_data"):
        data_note = (
            f"\n\nThe figure must plot EXACTLY these values (diagram_data — the source of truth). "
            f"For D1 FIDELITY, check the rendered bars/points/slices match them:\n"
            f"```json\n{json.dumps(q_data['diagram_data'], indent=2)}\n```"
        )
    human_content = [
        {"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": png_b64
        }},
        {"type": "text", "text": (
            f"The diagram illustrates this question:\n{q_data.get('text', '')}\n\n"
            f"TikZ source that produced the image above:\n```\n{q_data.get('TikZ_Code', '')}\n```"
            f"{data_note}\n\n"
            f"Score the figure on the D1-D5 rubric and respond in the required format."
        )},
    ]

    feedback_response = safe_invoke(_MODEL_SONNET, [
        SystemMessage(content=DIAGRAM_CRITIC_PROMPT),
        HumanMessage(content=human_content),
    ], max_tokens=700)
    feedback = feedback_response.content.strip()
    score, feedback = parse_diagram_score(feedback)

    if score >= DIAGRAM_PASS_SCORE:
        print(f"   ✅ Diagram approved! (score {score}/5)")
        return {"diagram_feedback": None, "diagram_score": score}
    else:
        fails = state.get("total_fail_count", 0) + 1
        print(f"   ⚠️  Diagram scored {score}/5 (visual refine {refines + 1}): {feedback[:160]}...")
        return {
            "diagram_feedback": feedback,
            "diagram_score":    score,
            "diagram_refines":  refines + 1,
            "total_fail_count": fails,
            "last_failure_type": "diagram",
        }

# ==========================================
# 3. ROUTING
# ==========================================
def route_after_compiler(state: QuestionState) -> str:
    # Deterministic math rejection (or compile/lint error, or no data) → regenerate,
    # skipping the factual critic entirely.
    if (state.get("compile_error")
            or (state.get("math_feedback") and state.get("last_failure_type") == "math")
            or not state.get("question_data")):
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
        # Diagrams get a dedicated refinement budget — keep refining the figure
        # rather than bailing at MAX_RETRIES. The DIAGRAM_MAX_REFINES cap (plus the
        # pivot-to-simpler/text logic in the generator) bounds the loop.
        if state.get("diagram_refines", 0) >= DIAGRAM_MAX_REFINES:
            print(f"🛑 Diagram refinement budget exhausted ({DIAGRAM_MAX_REFINES}).")
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
    subject,
    topic,
    sub_topic,
    n_per_level:      int,
    k_iterations:     int,
    difficulty_levels: List[int],
    syllabus_file:    str,
    output_file:      str,
    diagram_mode:     str = "auto",
    on_question=None,            # callback(q_data, banked_count) after each question is banked
    on_progress=None,           # callback(done_slots, total_slots, label) per slot
    max_rounds_per_slot: int = 0,  # 0 = unlimited; else give up on a slot after N rounds
    should_stop=None,           # callable -> bool; if True, stop the run gracefully
):
    # Build exam-specific prompts once at startup
    sys_prompt  = build_system_prompt(exam)
    crit_prompt = build_critic_prompt(exam)
    exam_cat    = _exam_category(exam)

    diagram_mode = (diagram_mode or "auto").lower()
    if diagram_mode not in ("auto", "force", "never"):
        print(f"⚠️  Unknown DIAGRAM_MODE '{diagram_mode}' — falling back to 'auto'")
        diagram_mode = "auto"

    print("\n🚀 Starting Question Bank Pipeline...")
    print(f"   Exam        : {exam}  [{exam_cat}]")
    print(f"   Subject     : {subject}")
    print(f"   Topic       : {topic}")
    print(f"   Sub-topic   : {sub_topic}")
    print(f"   N/level     : {n_per_level}  (questions per difficulty per iteration)")
    print(f"   K iterations: {k_iterations}")
    print(f"   Difficulties: {difficulty_levels}")
    _dm_desc = {"auto": "model decides per question",
                "force": "FORCE a figure on every question",
                "never": "NEVER produce a figure (text-only)"}[diagram_mode]
    print(f"   Diagram Mode: {diagram_mode}  ({_dm_desc})")
    print(f"   Haiku       : {_MODEL_HAIKU  or '⚠️  NOT SET — falls back to Sonnet'}")
    print(f"   Sonnet      : {_MODEL_SONNET}")
    print(f"   Opus        : {_MODEL_OPUS}")
    print(f"   Critics     : Factual=Sonnet | Diagram=Sonnet+Vision rubric (image+source, score≥{DIAGRAM_PASS_SCORE}/5)")
    print(f"   Budget      : {MAX_RETRIES} text attempts/round | {DIAGRAM_MAX_REFINES} visual refines/figure")
    print(f"   Renderer    : {RENDERER_URL}  (server-side SVG+PNG @ {RENDER_DPI} dpi)")
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

    # Per-subtopic structural patterns already banked, so we can push the generator
    # toward NEW structures instead of the same template with different numbers.
    from collections import defaultdict
    used_structures: Dict[str, List[str]] = defaultdict(list)
    for q in master_question_bank:
        st_key = (q.get("metadata", {}) or {}).get("sub_topic", "")
        sig = structure_fingerprint(q)
        if sig:
            used_structures[st_key].append(sig)

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

    # Total number of question slots (for progress reporting).
    total_slots = len(target_nodes) * len(difficulty_levels) * k_iterations * n_per_level
    done_slots = 0
    stopped = False

    # Main loop: subtopic → difficulty → iteration × n_per_level
    for subj, tpc, st in target_nodes:
        if stopped:
            break
        label = f"{subj} → {tpc} → {st}" if tpc else f"{subj} → {st}"
        print(f"\n{'='*58}")
        print(f"🎯  {label}")
        print(f"{'='*58}")

        for difficulty in difficulty_levels:
            if stopped:
                break
            for k in range(1, k_iterations + 1):
                if stopped:
                    break
                for n in range(1, n_per_level + 1):
                    if should_stop is not None and should_stop():
                        print("🛑 Stop requested — ending run gracefully.")
                        stopped = True
                        break
                    slot_label = f"Level {difficulty} | Iter {k}/{k_iterations} | Q {n}/{n_per_level}"
                    print(f"\n👉  {slot_label}")

                    slug      = re.sub(r'[^A-Z0-9]', '', st.upper())[:10]
                    exam_slug = re.sub(r'[^A-Z0-9]', '', exam.upper())[:6]

                    round_num      = 0
                    total_attempts = 0
                    banked         = False
                    tried_concepts: List[str] = []

                    while not banked:
                        if should_stop is not None and should_stop():
                            stopped = True
                            break
                        if max_rounds_per_slot and round_num >= max_rounds_per_slot:
                            print(f"   ⏭️  Slot gave up after {round_num} rounds "
                                  f"(max_rounds_per_slot={max_rounds_per_slot}). Moving on.")
                            break
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
                        _dm_hint = {
                            "auto":  "Auto — reason whether a diagram genuinely aids understanding "
                                     "BEFORE setting Requires_Diagram.",
                            "force": "Force — a figure is REQUIRED for this question "
                                     "(Requires_Diagram=true + valid TikZ_Code).",
                            "never": "Never — this question MUST be text-only "
                                     "(Requires_Diagram=false, TikZ_Code=null).",
                        }[diagram_mode]
                        request_parts += [
                            f"- Sub-topic: {st}",
                            f"- Difficulty Level: {difficulty} / 5",
                            f"- Question type preference: {_qtype_hint(difficulty)}",
                            f"- Diagram_Mode: {_dm_hint}",
                        ]

                        # VARIETY: discourage repeating structures already banked for this
                        # sub-topic (same template, different numbers). Show the recent
                        # structural signatures and ask for a genuinely different form.
                        recent_structs = used_structures.get(st, [])
                        if recent_structs:
                            # de-dup preserving order, keep the most recent few
                            seen_s, uniq = set(), []
                            for sig in reversed(recent_structs):
                                if sig not in seen_s:
                                    seen_s.add(sig); uniq.append(sig)
                                if len(uniq) >= 6:
                                    break
                            request_parts.append(
                                "- VARIETY (IMPORTANT): the following question STRUCTURES are "
                                "already banked for this sub-topic. Do NOT just reuse the same "
                                "template with different numbers — change the FORM of the question "
                                "(different operations, question phrasing, or solving approach):\n"
                                + "\n".join(f"    • {s}" for s in uniq)
                            )
                        request = "\n".join(request_parts) + extra_hint

                        print(f"   🔁 Round {round_num}")

                        image_dir = os.path.splitext(output_file)[0]

                        initial_state: QuestionState = {
                            "request_prompt":    request,
                            "forced_id":         forced_id,
                            "system_prompt":     sys_prompt,
                            "critic_prompt":     crit_prompt,
                            "generation_count":  0,
                            "total_fail_count":  0,
                            "last_failure_type": "",
                            "image_dir": image_dir,
                            "raw_json_str":      None,
                            "question_data":     None,
                            "compile_error":     None,
                            "math_feedback":     None,
                            "diagram_feedback":  None,
                            "diagram_refines":   0,
                            "diagram_score":     0,
                            "png_b64":           None,
                            "final_image_path":  None,
                            "used_numbers":      list(used_numbers),
                            "diagram_mode":      diagram_mode,
                        }

                        try:
                            final_state = app.invoke(initial_state)
                        except Exception as e:
                            # A slot failing (e.g. network truly down after all retries)
                            # must NOT crash the whole run or lose already-banked work.
                            print(f"   ❌ Slot errored ({str(e)[:100]}). Skipping this slot, "
                                  f"continuing run...")
                            break  # abandon this slot, move to the next one
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
                                final_img = os.path.join(image_dir, f"{q_data['id']}.svg")
                                os.rename(tmp_img, final_img)
                                q_data["local_image_path"] = final_img

                            fp = numeric_fingerprint(q_data)
                            if fp:
                                used_numbers.append(fp)

                            sig = structure_fingerprint(q_data)
                            if sig:
                                used_structures[st].append(sig)

                            master_question_bank.append(q_data)
                            with open(output_file, "w") as f:
                                json.dump(master_question_bank, f, indent=2)

                            icon = "📐" if q_data.get("Requires_Diagram") else "📝"
                            print(
                                f"   💾 Banked {icon}: {q_data['id']} "
                                f"(round {round_num}, {total_attempts} total attempts)"
                            )
                            banked = True
                            if on_question is not None:
                                try:
                                    on_question(q_data, len(master_question_bank))
                                except Exception as cb_err:
                                    print(f"   ⚠️  on_question callback error: {cb_err}")

                        else:
                            used_in_run = final_state.get("generation_count", 0)
                            print(
                                f"   ⚠️  Round {round_num} exhausted "
                                f"({used_in_run} attempts). Retrying with fresh angle..."
                            )

                    # Slot finished (banked, gave up, or stopped) → report progress.
                    done_slots += 1
                    if on_progress is not None:
                        try:
                            on_progress(done_slots, total_slots, f"{label} | {slot_label}")
                        except Exception:
                            pass

    total = len(master_question_bank)
    status = "stopped early" if stopped else "done"
    print(f"\n✅ {status.capitalize()}. {total} questions in {output_file}")
    return master_question_bank


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
    EXAM = "Bank PO"

    # ── SCOPE SELECTION ────────────────────────────────────────────────────
    # Each of SUBJECT / TOPIC / SUB_TOPIC accepts:
    #   • "All"                → every option at that level
    #   • a single name        → just that one
    #   • a LIST of names      → only those (e.g. ["Tabular DI", "Pie Chart DI"])
    # Examples:
    #   SUBJECT="All", TOPIC="All", SUB_TOPIC="All"  → entire exam
    #   SUBJECT="Indian Polity", TOPIC="All", SUB_TOPIC="All"  → all Polity subtopics
    #   SUBJECT="Indian Polity", TOPIC="All", SUB_TOPIC="Fundamental Rights (Articles 12–35)"
    #   SUB_TOPIC=["Tabular DI", "Pie Chart DI"]  → only those two subtopics

    SUBJECT   = "Quantitative Aptitude & Data Interpretation"
    TOPIC     = "Arithmetic"   # TOPIC level in IBPS PO (under Quantitative Aptitude)
    SUB_TOPIC = "All"                   # "All" = iterate every subtopic (Bar/Line/Pie/Tabular/Caselet DI)

    # ── GENERATION CONFIG ──────────────────────────────────────────────────
    N_PER_LEVEL       = 5          # questions to bank per difficulty level per iteration
    K_ITERATIONS      = 1          # iterations (K=2 doubles the total questions)
    DIFFICULTY_LEVELS = [1,2,3,4,5]

    # ── DIAGRAM CONTROL ─────────────────────────────────────────────────────
    # Manual override of whether questions get a figure:
    #   "auto"  → the model decides per question (recommended; uses exam-specific judgment)
    #   "force" → EVERY question must have a figure (Requires_Diagram=true, enforced)
    #   "never" → NO question gets a figure; any figure the model adds is stripped (text-only)
    # Tip: set "never" for text-only topics (Arithmetic, English, GA) and "auto"/"force"
    # for chart topics (Data Interpretation) by running those scopes in separate passes.
    DIAGRAM_MODE = "auto"

    # ── FILES ──────────────────────────────────────────────────────────────
    SYLLABUS_FILE = "syllabus_maps.json"
    # Set OUTPUT_FILE explicitly per exam to avoid cross-exam collisions.
    # OUTPUT_FILE   = "_stats_question_bank.json"
    # OUTPUT_FILE = EXAM+"_"+SUBJECT+"_"+TOPIC+"_"+SUB_TOPIC+".json"

    # Render a scope selector (string OR list) into a short filename-safe label.
    def _scope_label(sel) -> str:
        if isinstance(sel, (list, tuple, set)):
            names = list(sel)
            if len(names) == 1:
                sel = names[0]
            else:
                return f"{len(names)}sel"   # e.g. "3sel" for a list of 3
        return str(sel)

    safe_name = "_".join([
        EXAM, _scope_label(SUBJECT), _scope_label(TOPIC), _scope_label(SUB_TOPIC)
    ]).replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")
    OUTPUT_FILE = f"{safe_name}.json"

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
        diagram_mode      = DIAGRAM_MODE,
    )