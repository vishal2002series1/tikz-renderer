import os
import re
import requests
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# ==========================================
# 0. THE SYSTEM PROMPT (V7.0)
# ==========================================
SYSTEM_PROMPT = """You are an expert LaTeX, TikZ, PGFPlots, and Circuitikz developer. Your task is to generate complex, high-quality, and physically/mathematically accurate STEM problems with accompanying diagrams.

You must strictly adhere to the following compilation rules to ensure the code renders flawlessly via pdflatex without human intervention:

1. DOCUMENT STRUCTURE & UI: 
   - Start exactly with \\documentclass[varwidth=21cm, border=5mm]{standalone}.
   - Wrap everything in \\begin{document} ... \\end{document}.
   - Write the problem statement, questions, solutions, and figure captions as standard LaTeX text OUTSIDE the tikzpicture environment. Use tcolorbox for UI styling. Never use TikZ to draw the layout of the page.

2. EXPLICIT LIBRARIES: Explicitly load every package used.
   - For UI: \\usepackage[many]{tcolorbox}
   - For 3D: \\usepackage{tikz-3dplot}
   - For circuits: \\usepackage{circuitikz} (NEVER draw circuit components manually).
   - For plotting: \\usepackage{pgfplots} and \\pgfplotsset{compat=1.18}.
   - TikZ Libraries: \\usetikzlibrary{positioning, calc, intersections, arrows.meta, backgrounds, patterns, decorations.pathmorphing}.

3. SPATIAL AWARENESS & MATH IN COORDINATES:
   - Never guess absolute (x,y) coordinates for floating annotations, text labels, or ray-tracing intersections. Use positioning or calc to ensure physical accuracy.
   - IMPORTANT: If you perform math operations inside a TikZ coordinate, you MUST wrap the math in curly braces. E.g., Use ({5/2}, 0) NOT (5/2, 0).
   - For mechanical springs, use decoration={coil}.

4. VISUAL READABILITY & CONTRAST:
   - All text, labels, and captions must be highly legible. Do not use low-contrast colors for typography.

5. SCALE & DIMENSION LIMITS:
   - TeX will crash with a "Dimension too large" error if coordinates exceed ~16000pt. Keep all raw TikZ coordinates between -15 and +15.
   - Beware of rotated scopes. Ensure gravity vectors point absolutely downward relative to the page.

6. VECTORS & ARROWS: 
   - Always use the arrows.meta library for vectors. Set a global style like \\begin{tikzpicture}[>={Stealth[scale=1.2]}] so arrows are clearly visible.

7. NODE LR-MODE & LINE BREAKS:
   - If a \\node contains line breaks, you MUST provide a text width parameter.
   - NEVER use empty line breaks anywhere in the document (causes a fatal crash). Use \\\\[1em] or blank lines for spacing.

8. 3D PAINTER'S ALGORITHM: 
   - Draw background elements first, then midground, then foreground.

9. OUTPUT FORMAT: Output ONLY the raw, compilable LaTeX code inside a single markdown latex code block. No conversational filler.
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
    match = re.search(r"```(latex|tex|)[ \t]*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
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
        prompt_text = f"Create a STEM diagram/problem for the following request:\n<request>\n{user_prompt}\n</request>"

    llm = ChatBedrock(
        model_id="us.anthropic.claude-sonnet-4-6",
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        model_kwargs={"max_tokens": 20000}
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
        model_id="us.anthropic.claude-sonnet-4-6",
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        model_kwargs={"max_tokens": 1024}
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
    print("\n🚀 Starting Autonomous Diagram Pipeline...")
    
    # 1. Ask the user for their request via the CLI
    print("\n📝 Enter your diagram request (Type your prompt and press Enter):")
    user_input = input("> ")

    # Exit if the user just pressed Enter without typing anything
    if not user_input.strip():
        print("No prompt provided. Exiting.")
        exit()

    # 2. Pass the user's input into the LangGraph state
    initial_state: DiagramState = {
        "user_prompt": user_input,
        "generation_count": 0,
        "current_latex": None,
        "compile_error": None,
        "visual_feedback": None,
        "final_image_path": None
    }
    
    print("\n⏳ Processing your request. Please wait...")

    # 3. Stream the events as the graph executes
    for output in app.stream(initial_state):
        for node_name, state_update in output.items():
            print(f"\n--- Node finished: [{node_name}] ---")
            for key, value in state_update.items():
                if value is not None:
                    # Print a clean, single-line preview of the state updates
                    preview = str(value)[:120].replace("\n", " ")
                    print(f"  {key}: {preview}")
                    
    print("\n🎉 Pipeline Complete! Check 'output_diagram.svg' for the final result.")