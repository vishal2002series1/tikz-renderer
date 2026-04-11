import os
import requests
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langchain_aws import ChatBedrock

# Load environment variables from .env
load_dotenv()

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
# 2. DEFINE THE COMPILER NODE
# ==========================================
def compile_latex_node(state: DiagramState) -> dict:
    """
    Takes the generated LaTeX from the state, sends it to the Next.js renderer,
    and returns either a success state (with image path) or an error state.
    """
    latex_code = state.get("current_latex")
    if not latex_code:
        return {"compile_error": "No LaTeX code found in state to compile."}

    renderer_url = os.getenv("RENDERER_URL", "http://localhost:3002/api/render")
    print(f"🔄 [Compiler] Sending code to Next.js server at {renderer_url}...")

    try:
        response = requests.post(renderer_url, json={"code": latex_code})
        
        if response.status_code == 200:
            # Success! Save the SVG locally.
            output_path = "output_diagram.svg"
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"✅ [Compiler] Success! Diagram saved to {output_path}")
            # Clear any previous errors and update the image path
            return {"compile_error": None, "final_image_path": output_path}
            
        else:
            # Compilation failed. Extract the error from Next.js.
            error_data = response.json()
            error_msg = error_data.get("error", "Unknown compilation error.")
            print(f"❌ [Compiler] LaTeX Error: {error_msg[:100]}...") # Print just the start of the error
            return {"compile_error": error_msg, "final_image_path": None}
            
    except Exception as e:
        print(f"❌ [Compiler] Request failed: {str(e)}")
        return {"compile_error": f"Failed to connect to Next.js server: {str(e)}"}

# ==========================================
# 3. ENVIRONMENT SANITY CHECKS
# ==========================================
if __name__ == "__main__":
    print("--- Testing Environment ---")
    
    # Check 1: AWS Bedrock Connection
    try:
        print("Testing AWS Bedrock connection...")
        # Using Claude 3.5 Sonnet as the default model (excellent at coding and vision)
        llm = ChatBedrock(
            model_id="us.anthropic.claude-sonnet-4-6", 
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        )
        res = llm.invoke("Say 'Bedrock is working!'")
        print(f"✅ [Bedrock] {res.content}")
    except Exception as e:
        print(f"❌ [Bedrock] Connection failed. Check your AWS CLI profile: {e}")

    # Check 2: Next.js Compiler Tool
    print("\nTesting Next.js Compiler Tool...")
    test_state: DiagramState = {
        "user_prompt": "Test",
        "generation_count": 0,
        "current_latex": "\\documentclass[tikz, border=2mm]{standalone}\\begin{document}\\begin{tikzpicture}\\draw[fill=blue] (0,0) circle (1);\\end{tikzpicture}\\end{document}",
        "compile_error": None,
        "visual_feedback": None,
        "final_image_path": None
    }
    
    # Make sure your Next.js server is running on port 3002 before this runs!
    result = compile_latex_node(test_state)
    print(f"Compiler Result Update: {result}")