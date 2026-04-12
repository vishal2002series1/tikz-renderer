import os
import re
import json
import requests
from typing import TypedDict, Optional, Dict, Any
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.messages import SystemMessage, HumanMessage
from botocore.config import Config

load_dotenv()

# ==========================================
# 0. THE INTELLIGENT SYSTEM PROMPT
# ==========================================
SYSTEM_PROMPT = """You are an expert SSC CGL exam question setter and a LaTeX/TikZ expert. 

Your task is to generate high-quality, exam-accurate questions. 
You must decide if the question requires a geometric or logical diagram. 
- If the user specifies "Diagram_Mode: Force True", you MUST create a visual question.
- If "Diagram_Mode: Force False", you MUST NOT use a diagram.
- If "Diagram_Mode: Auto", use your intelligence. (e.g., Geometry, Venn Diagrams, and Non-Verbal Reasoning usually need diagrams. Arithmetic usually does not).

Output your response STRICTLY as a valid JSON object. 
If a diagram is needed, set "Requires_Diagram": true, and put the FULL, compilable LaTeX/TikZ code in "TikZ_Code".

The TikZ_Code MUST adhere to these rules:
1. Start with \\documentclass[varwidth=21cm, border=5mm]{standalone} and wrap in \\begin{document}...\\end{document}.
2. Explicitly load \\usepackage{tikz} and any needed libraries (e.g., \\usetikzlibrary{calc, positioning, angles, quotes}).
3. For text readability on lines, use `fill=white`.
4. Keep coordinates between -15 and +15 to avoid "Dimension too large" errors.

JSON SCHEMA:
{
  "id": "ssc_cgl_geo_001",
  "text": "The question text. Use $ for inline math and $$ for block math. Mention 'the given figure' if a diagram is included.",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "correct_answer": "A",
  "explanation": "Detailed step-by-step solution.",
  "Requires_Diagram": true or false,
  "TikZ_Code": "\\documentclass[varwidth=21cm, border=5mm]{standalone}\\usepackage{tikz}\\begin{document}\\begin{tikzpicture}...\\end{tikzpicture}\\end{document}" or null,
  "metadata": {"exam": "SSC CGL", "subject": "...", "topic": "...", "sub_topic": "...", "difficulty_level": 3}
}
"""

def extract_json(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(json|)[ \t]*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(2).strip() # CHANGED TO 2
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("\n", 1)[0]
    return text.strip()

# ==========================================
# 1. THE INTELLIGENT GENERATOR
# ==========================================
def generate_and_compile_question(user_request: str):
    print("\n🧠 [Generator] Thinking and writing question...")
    
    llm = ChatBedrock(
        model_id=os.getenv("Model_ID_Sonnet", "us.anthropic.claude-sonnet-4-6"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        model_kwargs={"max_tokens": 8192},
        config=Config(read_timeout=300)
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Generate the exam question based on the following request:\n<request>\n{user_request}\n</request>")
    ]

    response = llm.invoke(messages)
    raw_json_str = extract_json(response.content)
    
    try:
        question_data = json.loads(raw_json_str)
        print("✅ [Generator] Valid JSON successfully generated!")
    except json.JSONDecodeError as e:
        print(f"❌ [Generator] Failed to parse JSON: {e}")
        print(raw_json_str)
        return

    # Print the Question Text
    print(f"\n📝 Question: {question_data['text']}")
    print(f"📊 Diagram Required: {question_data['Requires_Diagram']}")
    # Print the Full Question Details
    print(f"\n📝 Question: {question_data['text']}")
    print(f"🔹 Options: {json.dumps(question_data['options'], indent=2)}")
    print(f"✅ Correct Answer: {question_data['correct_answer']}")
    print(f"💡 Explanation: {question_data['explanation']}")
    print(f"📊 Diagram Required: {question_data['Requires_Diagram']}")

    # ==========================================
    # 2. THE CONDITIONAL COMPILER
    # ==========================================
    if question_data.get("Requires_Diagram") and question_data.get("TikZ_Code"):
        print("\n🎨 [Compiler] Diagram detected! Sending TikZ code to Next.js server...")
        latex_code = question_data["TikZ_Code"]
        renderer_url = os.getenv("RENDERER_URL", "http://localhost:3002/api/render")
        
        try:
            render_response = requests.post(renderer_url, json={"code": latex_code})
            if render_response.status_code == 200:
                output_path = "visual_test_diagram.svg"
                with open(output_path, "wb") as f:
                    f.write(render_response.content)
                print(f"✅ [Compiler] Success! Diagram saved to {output_path}")
            else:
                error_data = render_response.json()
                print(f"❌ [Compiler] Compilation Failed:\n{error_data.get('error')[:300]}...")
        except Exception as e:
            print(f"❌ [Compiler] Request failed: {str(e)}")
    elif question_data.get("Requires_Diagram"):
        print("⚠️ [Warning] Requires_Diagram is true, but no TikZ_Code was provided!")

# ==========================================
# 3. RUN THE TEST
# ==========================================
if __name__ == "__main__":
    # Let's test a Geometry problem and FORCE the diagram to True
    test_request = """
- Subject: Tier 1 – General Intelligence & Reasoning
- Topic: Verbal Reasoning
- Subtopic: Direction & Distance
- Difficulty Level: 3 / 5
- Diagram_Mode: Auto
"""
    generate_and_compile_question(test_request)

