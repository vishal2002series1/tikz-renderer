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

Output your response STRICTLY as a valid JSON object. 
If a diagram is generated (either for the question or the explanation), set "Requires_Diagram": true, and put the FULL, compilable LaTeX/TikZ code in "TikZ_Code".

The TikZ_Code MUST adhere to these rules:
1. Start with \\documentclass[varwidth=21cm, border=5mm]{standalone} and wrap in \\begin{document}...\\end{document}.
2. Explicitly load \\usepackage{tikz} and any needed libraries.
3. For text readability on lines, use `fill=white`.
4. Keep coordinates between -15 and +15 to avoid "Dimension too large" errors.

JSON SCHEMA:
{
  "id": "<generate_a_unique_id_here>",
  "text": "The question text. Use $ for inline math and $$ for block math. Mention 'the given figure' if a diagram is included for the question.",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "correct_answer": "A",
  "explanation": "Detailed step-by-step solution. Mention 'refer to the generated diagram' if the diagram is meant for the solution.",
  "Requires_Diagram": true or false,
  "TikZ_Code": "LaTeX code here" or null,
  "metadata": {
    "exam": "SSC CGL",
    "subject": "<subject>",
    "topic": "<topic>",
    "sub_topic": "<sub_topic>",
    "difficulty_level": <difficulty>
  }
}
"""

# ==========================================
# 1. STATE & HELPERS
# ==========================================
class QuestionState(TypedDict):
    request_prompt: str
    generation_count: int
    raw_json_str: Optional[str]
    question_data: Optional[Dict[str, Any]]
    compile_error: Optional[str]
    critic_feedback: Optional[str]
    final_image_path: Optional[str]

def extract_json(text: str) -> str:
    text = text.strip()
    match = re.search(r"'''(json|)[ \t]*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
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
            f"Please fix the LaTeX/TikZ code and return the FULL, corrected JSON."
        )
    elif state.get("critic_feedback") and prev_json:
        print("🧠 [Generator] Fixing Critic Feedback...")
        prompt_text += (
            f"\n\nYou previously generated this JSON:\n```json\n{prev_json}\n```\n\n"
            f"However, the QA Critic rejected it with this feedback:\n<feedback>\n{state['critic_feedback']}\n</feedback>\n"
            f"Please fix the mathematical or visual errors and return the FULL, corrected JSON."
        )

    llm = ChatBedrock(model_id=MODEL_ID, region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"), model_kwargs={"max_tokens": 8192}, config=Config(read_timeout=300))
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt_text)])
    
    raw_json = extract_json(response.content)
    try:
        q_data = json.loads(raw_json)
        if "id" not in q_data or q_data["id"] == "<generate_a_unique_id_here>":
            q_data["id"] = f"ssc_cgl_{uuid.uuid4().hex[:8]}"
        return {"raw_json_str": raw_json, "question_data": q_data, "generation_count": gen_count + 1}
    except json.JSONDecodeError as e:
        return {"compile_error": f"Invalid JSON generated: {e}", "generation_count": gen_count + 1}

def compiler_node(state: QuestionState) -> dict:
    q_data = state.get("question_data")
    if not q_data or not q_data.get("Requires_Diagram"):
        return {"compile_error": None, "final_image_path": None} 

    print(f"\n🎨 [Compiler] Diagram detected. Rendering via Next.js...")
    try:
        res = requests.post(RENDERER_URL, json={"code": q_data["TikZ_Code"]})
        if res.status_code == 200:
            img_name = f"{q_data['id']}.svg"
            img_path = os.path.join("local_images", img_name)
            os.makedirs("local_images", exist_ok=True)
            with open(img_path, "wb") as f:
                f.write(res.content)
            print(f"✅ [Compiler] Success! Saved as {img_name}")
            return {"compile_error": None, "final_image_path": img_path}
        else:
            return {"compile_error": res.json().get("error", "Unknown compilation error."), "final_image_path": None}
    except Exception as e:
        return {"compile_error": f"Renderer connection failed: {e}", "final_image_path": None}

def critic_node(state: QuestionState) -> dict:
    q_data = state.get("question_data")
    print("\n🔍 [Dual-Critic] Reviewing math, logic, and layout...")

    critic_prompt = f"""You are a strict QA Reviewer for SSC CGL exams.
Review this generated question:
{json.dumps(q_data, indent=2)}

Perform a DUAL-CRITIC check:
1. MATHEMATICAL: Solve the problem yourself. Does the math exactly match the correct_answer option?
2. CHEATING CHECK: Did the generator include a diagram in the question text that gives away the answer to a word problem? 
3. VISUAL (If Requires_Diagram is true): Look at the TikZ_Code. Are there any absolute coordinates that will cause overlaps? Are arrows routed cleanly?

If the question is flawless, reply with ONLY the word: PASS.
If there are ANY errors, provide a brief, specific list of what needs to be fixed."""

    llm = ChatBedrock(model_id=MODEL_ID, region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"), model_kwargs={"max_tokens": 1024}, config=Config(read_timeout=300))
    feedback = llm.invoke([HumanMessage(content=critic_prompt)]).content.strip()

    if "PASS" in feedback.upper():
        print("✅ [Dual-Critic] Approved!")
        return {"critic_feedback": None}
    else:
        print(f"⚠️ [Dual-Critic] Rejected: {feedback[:100]}...")
        return {"critic_feedback": feedback}

# ==========================================
# 3. ROUTING & GRAPH SETUP
# ==========================================
def route_compiler(state: QuestionState) -> str:
    if state.get("compile_error"):
        return END if state["generation_count"] >= MAX_RETRIES else "generator_node"
    return "critic_node"

def route_critic(state: QuestionState) -> str:
    if state.get("critic_feedback"):
        return END if state["generation_count"] >= MAX_RETRIES else "generator_node"
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

    # We manually define the two highly-visual subtopics for this test
    TARGET_NODES = [
        ("Tier 1 – Quantitative Aptitude", "Advanced Mathematics", "Geometry – Triangles, Circles, Quadrilaterals, Coordinate Geometry"),
        ("Tier 1 – General Intelligence & Reasoning", "Non-Verbal Reasoning", "Venn Diagrams")
    ]

    for tier_subject, topic, sub_topic in TARGET_NODES:
        print(f"\n=========================================")
        print(f"🎯 Generating questions for: {sub_topic}")
        print(f"=========================================")

        # Generate from Level 1 to 5
        for difficulty in range(1, 6):
            # Generate 2 questions per difficulty level
            for iteration in range(1, 3):
                print(f"\n👉 Processing {sub_topic} | Level {difficulty} | Question {iteration}/2")
                
                request = f"""
- Subject: {tier_subject}
- Topic: {topic}
- Subtopic: {sub_topic}
- Difficulty Level: {difficulty} / 5
- Diagram_Mode: Auto (Follow system prompt rules strictly!)
"""
                state = {
                    "request_prompt": request, "generation_count": 0, "raw_json_str": None,
                    "question_data": None, "compile_error": None, "critic_feedback": None, "final_image_path": None
                }

                final_state = app.invoke(state)
                
                if final_state.get("question_data") and not final_state.get("compile_error") and not final_state.get("critic_feedback"):
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
