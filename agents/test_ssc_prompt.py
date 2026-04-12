import os
import re
import requests
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from botocore.config import Config

# Load environment variables
load_dotenv()

# ==========================================
# 0. THE SYSTEM PROMPT (SSC CGL JSON)
# ==========================================
SYSTEM_PROMPT = """You are an expert SSC CGL (Staff Selection Commission - Combined Graduate Level) exam question setter. Your task is to generate high-quality, exam-accurate questions based on specific syllabus topics.

Output your response STRICTLY as a valid JSON object matching the following structure. Do not include any markdown formatting outside the JSON object, conversational text, or explanations outside the JSON fields.

{
  "id": "A unique string ID (e.g., ssc_cgl_quant_001)",
  "text": "The question text. Use $ for inline math (e.g., $x=5$) and $$ for block math.",
  "options": {
    "A": "Option 1",
    "B": "Option 2",
    "C": "Option 3",
    "D": "Option 4"
  },
  "correct_answer": "A, B, C, or D",
  "explanation": "A detailed, step-by-step solution. Use $ for math.",
  "Requires_Diagram": false,
  "TikZ_Code": null,
  "metadata": {
    "exam": "SSC CGL",
    "subject": "The requested subject",
    "topic": "The requested topic",
    "sub_topic": "The requested subtopic",
    "difficulty_level": 3
  }
}

Guidelines for Difficulty Level 3/5 (Exam Level):
- The question should match the exact difficulty, tone, and time-complexity of actual SSC CGL Tier-1 exams.
- It should require 1 to 2 conceptual steps to solve, taking an average student about 45-60 seconds.
- Avoid overly simple direct-formula questions, but do not make the calculation unrealistically tedious.
"""

# ==========================================
# 1. DEFINE THE LANGGRAPH STATE
# ==========================================
class DiagramState(TypedDict):
    user_prompt: str
    generation_count: int
    current_latex: Optional[str]
    compile_error: Optional[str]
    visual_feedback: Optional[str]
    final_image_path: Optional[str]

# ==========================================
# 2. HELPER: EXTRACT LATEX
# ==========================================
def extract_latex(text: str) -> str:
    """Extracts LaTeX code from markdown code blocks robustly."""
    text = text.strip()
 
    # Regex to find everything between ```latex / ```tex / ``` and closing ```
    match = re.search(r"```(json|)[ \t]*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(2).strip()
 
    # Fallback: Manual string stripping just in case regex misses
    if text.startswith("```"):
        # Split off the first line (which has the opening backticks)
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        # Split off the last line (which has the closing backticks)
        text = text.rsplit("\n", 1)[0]
 
    return text.strip()

# ==========================================
# 3. NODE: GENERATOR AGENT
# ==========================================
def generator_node(state: DiagramState) -> dict:
    user_prompt = state.get("user_prompt", "")
    current_latex = state.get("current_latex")
    compile_error = state.get("compile_error")
    visual_feedback = state.get("visual_feedback")
    gen_count = state.get("generation_count", 0)

    print(f"\n🧠 [Generator] Generation attempt {gen_count + 1}...")

    # Determine the context of the prompt
    if compile_error:
        print("🧠 [Generator] Mode: Fixing Compiler Error...")
        prompt_text = (
            f"You previously generated this LaTeX code:\n\n{current_latex}\n\n"
            f"However, it failed to compile with the following pdflatex error:\n"
            f"<error>\n{compile_error}\n</error>\n\n"
            f"Please fix the error and provide the corrected, full, and compilable LaTeX code."
        )
    elif visual_feedback:
        print("🧠 [Generator] Mode: Fixing Visual Layout...")
        prompt_text = (
            f"You previously generated this LaTeX code:\n\n{current_latex}\n\n"
            f"It compiled successfully, but the visual reviewer provided the following feedback:\n"
            f"<feedback>\n{visual_feedback}\n</feedback>\n\n"
            f"Please adjust the TikZ/LaTeX code to address these visual issues while keeping physical accuracy."
        )
    else:
        print("🧠 [Generator] Mode: Initial Creation...")
        prompt_text = f"Generate the exam question based on the following request:\n<request>\n{user_prompt}\n</request>"

    llm = ChatBedrock(
        model_id=os.getenv("Model_ID","us.anthropic.claude-sonnet-4-6"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        model_kwargs={"max_tokens": 20000},
        config=Config(read_timeout=300)
        
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt_text)
    ]

    response = llm.invoke(messages)
    raw_latex = extract_latex(response.content)

    print("🧠 [Generator] LaTeX generated successfully.")

    return {
        "current_latex": raw_latex,
        "generation_count": gen_count + 1
    }

# ==========================================
# 4. NODE: COMPILER
# ==========================================
def compile_latex_node(state: DiagramState) -> dict:
    latex_code = state.get("current_latex")
    if not latex_code:
        return {"compile_error": "No LaTeX code found in state to compile."}

    renderer_url = os.getenv("RENDERER_URL", "http://localhost:3002/api/render")
    print(f"\n🔄 [Compiler] Sending code to Next.js server at {renderer_url}...")

    try:
        response = requests.post(renderer_url, json={"code": latex_code})

        if response.status_code == 200:
            output_path = "output_diagram.svg"
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"✅ [Compiler] Success! Diagram saved to {output_path}")
            return {"compile_error": None, "final_image_path": output_path}
        else:
            error_data = response.json()
            error_msg = error_data.get("error", "Unknown compilation error.")
            print(f"❌ [Compiler] LaTeX Error: {error_msg[:100]}...")
            return {"compile_error": error_msg, "final_image_path": None}

    except Exception as e:
        print(f"❌ [Compiler] Request failed: {str(e)}")
        return {"compile_error": f"Failed to connect to Next.js server: {str(e)}"}

# ==========================================
# 5. NODE: THE CRITIC AGENT
# ==========================================
def critic_node(state: DiagramState) -> dict:
    current_latex = state.get("current_latex")
    user_prompt = state.get("user_prompt")
 
    print("\n🔍 [Critic] Reviewing the generated diagram logic...")
 
    critic_prompt = f"""You are a strict QA Reviewer for LaTeX/TikZ diagrams.
The user originally requested: "{user_prompt}"
 
Here is the generated code that compiled successfully:
```latex
{current_latex}
```
 
Please review the code for the following:
 
1. Did it actually fulfill the user's request accurately?
2. Did it use absolute coordinates for floating text/nodes (which causes overlaps)?
3. Are there any missing `text width` parameters on nodes with long text?
 
If the code looks solid, physically accurate, and well-structured, reply with ONLY the word: PASS.
If there are logical or layout issues, provide a brief, specific list of what needs to be fixed.
Do NOT rewrite the code, just give feedback."""
 
    llm = ChatBedrock(
        model_id=os.getenv("Model_ID","us.anthropic.claude-sonnet-4-6"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        model_kwargs={"max_tokens": 1024},
        config=Config(read_timeout=300)
    )
 
    response = llm.invoke([HumanMessage(content=critic_prompt)])
    feedback = response.content.strip()
 
    if "PASS" in feedback.upper():
        print("✅ [Critic] Diagram approved! No visual/logical feedback.")
        return {"visual_feedback": None}
    else:
        print(f"⚠️ [Critic] Issues found: {feedback[:100]}...")
        return {"visual_feedback": feedback}
 
 
# ==========================================
# 6. GRAPH ROUTING LOGIC
# ==========================================
def route_after_compiler(state: DiagramState) -> str:
    """Decides where to go after compilation."""
    if state.get("compile_error"):
        if state.get("generation_count", 0) >= 3:
            print("🛑 [Router] Max retries reached for compile errors. Halting.")
            return END
        return "generator_node"
    return "critic_node"
 
 
def route_after_critic(state: DiagramState) -> str:
    """Decides where to go after the critic reviews it."""
    if state.get("visual_feedback"):
        if state.get("generation_count", 0) >= 3:
            print("🛑 [Router] Max retries reached for critic feedback. Halting.")
            return END
        return "generator_node"
    return END
 
 
# ==========================================
# 7. ASSEMBLE THE LANGGRAPH
# ==========================================
print("\n⚙️ Building the LangGraph State Machine...")
workflow = StateGraph(DiagramState)
 
# Add Nodes
workflow.add_node("generator_node", generator_node)
workflow.add_node("compile_latex_node", compile_latex_node)
workflow.add_node("critic_node", critic_node)
 
# Define Edges
workflow.set_entry_point("generator_node")
workflow.add_edge("generator_node", "compile_latex_node")
 
# Add Conditional Edges
workflow.add_conditional_edges("compile_latex_node", route_after_compiler)
workflow.add_conditional_edges("critic_node", route_after_critic)
 
# Compile the graph
app = workflow.compile()
 
 
# ==========================================
# 8. RUN THE AUTONOMOUS PIPELINE
# ==========================================
if __name__ == "__main__":
    print("\n🚀 Testing SSC CGL Level 3 Prompt...")
    
    # The specific test request
    user_request = """Generate 1 question for the following SSC CGL syllabus node:
- Subject: Quantitative Aptitude
- Topic: Arithmetic
- Subtopic: Time, Speed & Distance
- Difficulty Level: 5 / 5"""

    test_state: DiagramState = {
        "user_prompt": user_request,
        "generation_count": 0,
        "current_latex": None,
        "compile_error": None,
        "visual_feedback": None,
        "final_image_path": None
    }
    
    # Run ONLY the generator node to see what the LLM outputs
    new_state = generator_node(test_state)
    
    print("\n--- Raw JSON Output ---")
    print(new_state["current_latex"]) # We are temporarily storing the JSON in the latex state variable