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
# 0. PROMPTS & CONFIG
# ==========================================
MAX_RETRIES = 10
RENDERER_URL = os.getenv("RENDERER_URL", "http://localhost:3002/api/render")
MODEL_ID = os.getenv("Model_ID", "us.anthropic.claude-sonnet-4-6")

SYSTEM_PROMPT = """You are an expert SSC CGL exam question setter and a LaTeX/TikZ expert. 

Your task is to generate high-quality, exam-accurate questions. 
Decide if the question requires a geometric or logical diagram. 
- Geometry, Venn Diagrams, and Non-Verbal Reasoning usually need diagrams. Arithmetic usually does not.

CRITICAL DIAGRAM RULE ("NO CHEATING"):
- ONLY include a diagram in the question text if it is strictly necessary to state the problem (e.g., "In the given figure..."). 
- DO NOT include a diagram in the question text if translating the word problem into a mental image is part of the challenge for the student.
- You MAY use diagrams in the "explanation" section to visually demonstrate the solution if it adds high educational value.

Output your response STRICTLY as a valid JSON object with NO markdown fences, no preamble, and no trailing text.
If a diagram is generated (either for the question or the explanation), set "Requires_Diagram": true, and put the FULL, compilable LaTeX/TikZ code in "TikZ_Code".

The TikZ_Code MUST adhere to these rules:
1. Start with \\documentclass[varwidth=21cm, border=5mm]{standalone} and wrap in \\begin{document}...\\end{document}.
2. Explicitly load \\usepackage{tikz} and any needed libraries.
3. For text readability on lines, use `fill=white`.
4. Keep coordinates between -15 and +15 to avoid "Dimension too large" errors.

JSON SCHEMA (output ONLY this, no markdown, no extra text):
{
  "id": "PLACEHOLDER_WILL_BE_OVERWRITTEN",
  "text": "The question text. Use $ for inline math and $$ for block math. Mention 'the given figure' if a diagram is included for the question.",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "correct_answer": "A",
  "explanation": "Detailed step-by-step solution. Mention 'refer to the generated diagram' if the diagram is meant for the solution.",
  "Requires_Diagram": true,
  "TikZ_Code": "LaTeX code here or null",
  "metadata": {
    "exam": "SSC CGL",
    "subject": "<subject>",
    "topic": "<topic>",
    "sub_topic": "<sub_topic>",
    "difficulty_level": 1
  }
}
"""

# ==========================================
# 1. STATE & HELPERS
# ==========================================
class QuestionState(TypedDict):
    request_prompt: str
    forced_id: str          # FIX 1: ID is set by orchestrator, not the LLM
    generation_count: int
    raw_json_str: Optional[str]
    question_data: Optional[Dict[str, Any]]
    compile_error: Optional[str]
    critic_feedback: Optional[str]
    final_image_path: Optional[str]


def extract_json(text: str) -> str:
    """
    FIX 2: Corrected regex — was mixing ''' with ```, and returning group(1) instead of group(2).
    Now correctly extracts JSON from ```json ... ``` blocks, or falls back to raw stripping.
    """
    text = text.strip()

    # Try to match ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?[ \t]*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()  # group(1) is the content, not the language tag

    # Fallback: strip leading/trailing ``` lines manually
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("\n", 1)[0]

    return text.strip()


# ==========================================
# 2. GRAPH NODES
# ==========================================
def generator_node(state: QuestionState) -> dict:
    gen_count = state.get("generation_count", 0)
    print(f"\n🧠 [Generator] Attempt {gen_count + 1}...")

    prompt_text = f"Generate the exam question based on the following request:\n<request>\n{state['request_prompt']}\n</request>"

    prev_json = state.get("raw_json_str")

    if state.get("compile_error") and prev_json:
        print("🧠 [Generator] Fixing Compiler Error...")
        prompt_text += (
            f"\n\nYou previously generated this JSON:\n```json\n{prev_json}\n```\n\n"
            f"However, the TikZ code failed to compile with this error:\n<error>\n{state['compile_error']}\n</error>\n"
            f"Please fix the LaTeX/TikZ code and return the FULL, corrected JSON. "
            f"Output ONLY the raw JSON — no markdown fences, no preamble."
        )
    elif state.get("critic_feedback") and prev_json:
        print("🧠 [Generator] Fixing Critic Feedback...")
        # FIX 3: Critic prompt now includes the computed correct answer to break the loop
        prompt_text += (
            f"\n\nYou previously generated this JSON:\n```json\n{prev_json}\n```\n\n"
            f"A QA Critic rejected it with this feedback:\n<feedback>\n{state['critic_feedback']}\n</feedback>\n\n"
            f"IMPORTANT: The critic has computed the correct answer above. "
            f"You MUST change your question's correct_answer and/or the option values to match the critic's computed result exactly. "
            f"Do NOT regenerate the same numbers — either fix the answer or choose completely different numbers. "
            f"Return the FULL, corrected JSON. Output ONLY raw JSON — no markdown fences, no preamble."
        )

    llm = ChatBedrock(
        model_id=MODEL_ID,
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        model_kwargs={"max_tokens": 8192},
        config=Config(read_timeout=300)
    )
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt_text)])

    raw_json = extract_json(response.content)
    try:
        q_data = json.loads(raw_json)

        # FIX 1: Always overwrite the LLM-generated ID with our forced deterministic ID
        q_data["id"] = state["forced_id"]

        return {
            "raw_json_str": json.dumps(q_data),  # keep in sync after id overwrite
            "question_data": q_data,
            "generation_count": gen_count + 1,
            "compile_error": None,   # clear stale errors on successful parse
            "critic_feedback": None,
        }
    except json.JSONDecodeError as e:
        print(f"❌ [Generator] JSON parse failed: {e}")
        return {
            "compile_error": f"Invalid JSON generated: {e}",
            "generation_count": gen_count + 1
        }


def compiler_node(state: QuestionState) -> dict:
    q_data = state.get("question_data")
    if not q_data or not q_data.get("Requires_Diagram") or not q_data.get("TikZ_Code"):
        return {"compile_error": None, "final_image_path": None}

    print(f"\n🎨 [Compiler] Diagram detected. Rendering via Next.js...")
    try:
        res = requests.post(RENDERER_URL, json={"code": q_data["TikZ_Code"]})
        if res.status_code == 200:
            # FIX 1: ID is already unique (set by orchestrator), so filename is always unique
            img_name = f"{q_data['id']}.svg"
            img_path = os.path.join("local_images", img_name)
            os.makedirs("local_images", exist_ok=True)
            with open(img_path, "wb") as f:
                f.write(res.content)
            print(f"✅ [Compiler] Success! Saved as {img_name}")
            return {"compile_error": None, "final_image_path": img_path}
        else:
            err = res.json().get("error", "Unknown compilation error.")
            print(f"❌ [Compiler] Error: {err[:120]}")
            return {"compile_error": err, "final_image_path": None}
    except Exception as e:
        print(f"❌ [Compiler] Connection failed: {e}")
        return {"compile_error": f"Renderer connection failed: {e}", "final_image_path": None}


def critic_node(state: QuestionState) -> dict:
    q_data = state.get("question_data")
    print("\n🔍 [Dual-Critic] Reviewing math, logic, and layout...")

    critic_prompt = f"""You are a strict QA Reviewer for SSC CGL exams.
Review this generated question:
{json.dumps(q_data, indent=2)}

Perform a DUAL-CRITIC check:

1. MATHEMATICAL: Solve the problem yourself from scratch, step by step. 
   Show your working. Does your computed answer EXACTLY match the value in the correct_answer option?
   If not, state what the correct answer SHOULD be with your full working.

2. CHEATING CHECK: Did the generator include a diagram in the question text that gives away the 
   answer to a word problem? (Diagrams are fine for geometry problems that say "In the given figure...")

3. VISUAL (only if Requires_Diagram is true): Look at the TikZ_Code. Are there absolute coordinates 
   that could cause overlaps? Are arrows routed cleanly?

If the question is COMPLETELY flawless on all counts, reply with ONLY the word: PASS
If there are ANY errors, provide a numbered list of specific issues. Always include your computed 
correct answer when there is a math mismatch."""

    llm = ChatBedrock(
        model_id=MODEL_ID,
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        model_kwargs={"max_tokens": 2048},  # increased for full working
        config=Config(read_timeout=300)
    )
    feedback = llm.invoke([HumanMessage(content=critic_prompt)]).content.strip()

    if "PASS" in feedback.upper():
        print("✅ [Dual-Critic] Approved!")
        return {"critic_feedback": None}
    else:
        print(f"⚠️ [Dual-Critic] Rejected: {feedback[:120]}...")
        return {"critic_feedback": feedback}


# ==========================================
# 3. ROUTING & GRAPH SETUP
# ==========================================
def route_compiler(state: QuestionState) -> str:
    if state.get("compile_error"):
        if state["generation_count"] >= MAX_RETRIES:
            print("🛑 [Router] Max retries hit on compile errors.")
            return END
        return "generator_node"
    return "critic_node"


def route_critic(state: QuestionState) -> str:
    if state.get("critic_feedback"):
        if state["generation_count"] >= MAX_RETRIES:
            print("🛑 [Router] Max retries hit on critic feedback.")
            return END
        return "generator_node"
    return END


workflow = StateGraph(QuestionState)
workflow.add_node("generator_node", generator_node)
workflow.add_node("compile_latex_node", compiler_node)
workflow.add_node("critic_node", critic_node)
workflow.set_entry_point("generator_node")
workflow.add_edge("generator_node", "compile_latex_node")
workflow.add_conditional_edges("compile_latex_node", route_compiler)
workflow.add_conditional_edges("critic_node", route_critic)
app = workflow.compile()


# ==========================================
# 4. ORCHESTRATOR / BANK SEEDER
# ==========================================
def run_seeder():
    print("\n🚀 Starting SSC CGL Seed Bank Pipeline (VISUAL MASS TEST)...")

    master_question_bank = []
    output_file = "ssc_cgl_question_bank.json"

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            master_question_bank = json.load(f)

    TARGET_NODES = [
        ("Tier 1 – Quantitative Aptitude", "Advanced Mathematics", "Geometry – Triangles, Circles, Quadrilaterals, Coordinate Geometry"),
        ("Tier 1 – General Intelligence & Reasoning", "Non-Verbal Reasoning", "Venn Diagrams")
    ]

    for tier_subject, topic, sub_topic in TARGET_NODES:
        print(f"\n=========================================")
        print(f"🎯 Generating questions for: {sub_topic}")
        print(f"=========================================")

        for difficulty in range(1, 6):
            for iteration in range(1, 3):
                print(f"\n👉 Processing {sub_topic} | Level {difficulty} | Question {iteration}/2")

                # FIX 1: Generate a truly unique, deterministic ID here — never trust the LLM for this
                forced_id = f"SSC_CGL_{sub_topic[:8].replace(' ', '').upper()}_{difficulty}_{iteration}_{uuid.uuid4().hex[:6]}"

                request = f"""
- Subject: {tier_subject}
- Topic: {topic}
- Subtopic: {sub_topic}
- Difficulty Level: {difficulty} / 5
- Diagram_Mode: Auto (Follow system prompt rules strictly!)
"""
                state = {
                    "request_prompt": request,
                    "forced_id": forced_id,   # passed into graph state
                    "generation_count": 0,
                    "raw_json_str": None,
                    "question_data": None,
                    "compile_error": None,
                    "critic_feedback": None,
                    "final_image_path": None,
                }

                final_state = app.invoke(state)

                if (
                    final_state.get("question_data")
                    and not final_state.get("compile_error")
                    and not final_state.get("critic_feedback")
                ):
                    if final_state.get("final_image_path"):
                        final_state["question_data"]["local_image_path"] = final_state["final_image_path"]

                    master_question_bank.append(final_state["question_data"])

                    with open(output_file, "w") as f:
                        json.dump(master_question_bank, f, indent=2)
                    print(f"💾 Successfully banked Question ID: {final_state['question_data']['id']}")
                else:
                    print(f"🛑 Failed to bank question after {MAX_RETRIES} attempts.")


if __name__ == "__main__":
    run_seeder()