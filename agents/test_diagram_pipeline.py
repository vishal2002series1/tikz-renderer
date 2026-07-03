"""
test_diagram_pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━
Offline test harness for the diagram pipeline — NO AWS / Bedrock calls.

Verifies the parts of generic_exam_question_bank.py that don't need an LLM:
  1. Deterministic TikZ linter         (lint_tikz)
  2. Chart data-first validation       (validate_diagram_data / is_chart_question)
  3. Diagram-score parsing             (parse_diagram_score)
  4. Live renderer round-trip          (render_tikz)  — only if the renderer is up

Run it any time you change the pipeline, before spending tokens on a full run:

    ./venv/bin/python test_diagram_pipeline.py

The renderer test is skipped automatically if RENDERER_URL is unreachable, so the
deterministic checks still run with no server. To include it, start the renderer first:

    npm run dev -- -p 3002          # in the project root (or set RENDERER_URL)
"""

import os
import sys
import base64

import generic_exam_question_bank as g

# ── tiny test framework ──────────────────────────────────────────────────────
_PASS = 0
_FAIL = 0

def check(name: str, condition: bool, detail: str = ""):
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  ✅ {name}")
    else:
        _FAIL += 1
        print(f"  ❌ {name}" + (f"\n       {detail}" if detail else ""))


# ── 1. TikZ linter ───────────────────────────────────────────────────────────
def test_lint():
    print("\n── TikZ linter ──")

    clean = (r"\documentclass[border=4mm]{standalone}\usepackage{tikz}"
             r"\begin{document}\begin{tikzpicture}\node (a) at (0,0){x};"
             r"\end{tikzpicture}\end{document}")
    check("clean TikZ has no problems", g.lint_tikz(clean) == [], str(g.lint_tikz(clean)))

    check("empty code flagged", g.lint_tikz("") != [])

    unbalanced = r"\begin{document}\begin{tikzpicture}\node (a) {x;\end{tikzpicture}\end{document}"
    check("unbalanced braces flagged",
          any("brace" in p for p in g.lint_tikz(unbalanced)))

    fit_before = (r"\documentclass{standalone}\usepackage{tikz}\usetikzlibrary{fit}"
                  r"\begin{document}\begin{tikzpicture}"
                  r"\node[fit=(a)(b)] (box){};"
                  r"\node (a) at (0,0){x};\node (b) at (1,0){y};"
                  r"\end{tikzpicture}\end{document}")
    check("fit-before-nodes flagged",
          any("fit references" in p for p in g.lint_tikz(fit_before)))

    fit_after = (r"\documentclass{standalone}\usepackage{tikz}\usetikzlibrary{fit}"
                 r"\begin{document}\begin{tikzpicture}"
                 r"\node (a) at (0,0){x};\node (b) at (1,0){y};"
                 r"\node[fit=(a)(b)] (box){};"
                 r"\end{tikzpicture}\end{document}")
    check("fit-after-nodes is clean",
          not any("fit references" in p for p in g.lint_tikz(fit_after)),
          str(g.lint_tikz(fit_after)))

    missing_lib = (r"\documentclass{standalone}\usepackage{tikz}"
                   r"\begin{document}\begin{tikzpicture}"
                   r"\node[above=of a] (z) at (2,0){x};\end{tikzpicture}\end{document}")
    check("missing \\usetikzlibrary{positioning} flagged",
          any("positioning" in p for p in g.lint_tikz(missing_lib)))

    bad_coord = (r"\documentclass{standalone}\usepackage{tikz}"
                 r"\begin{document}\begin{tikzpicture}"
                 r"\node (z) at (20,0){x};\end{tikzpicture}\end{document}")
    check("out-of-bounds coordinate flagged",
          any("coordinate" in p for p in g.lint_tikz(bad_coord)))


# ── 2. Chart data-first validation ───────────────────────────────────────────
def test_chart_validation():
    print("\n── Chart data-first validation ──")

    good = {
        "metadata": {"sub_topic": "Bar Graph DI", "topic": "Data Interpretation"},
        "text": "The bar graph shows loan applications by five branches. Refer to the chart. "
                "The bank approves 35% of all applications. What is the difference between years?",
        "diagram_data": {
            "chart_type": "bar",
            "x_labels": ["P", "Q", "R", "S", "T"],
            "series": [
                {"name": "2022", "values": [480, 360, 540, 420, 300]},
                {"name": "2023", "values": [560, 450, 480, 500, 390]},
            ],
            "unit": "applications",
        },
    }
    check("well-formed data-first chart is clean",
          g.validate_diagram_data(good) == [], str(g.validate_diagram_data(good)))
    check("is_chart_question true for DI sub-topic", g.is_chart_question(good))

    missing = {k: v for k, v in good.items() if k != "diagram_data"}
    check("chart missing diagram_data flagged",
          any("missing 'diagram_data'" in p for p in g.validate_diagram_data(missing)))

    leaky = dict(good)
    leaky["text"] = ("Branch P=480, Q=360, R=540, S=420, T=300 in 2022 and "
                     "560, 450, 480, 500, 390 in 2023. Approve 35%. Difference?")
    check("redundant stem (restates all values) flagged",
          any("ANTI-REDUNDANCY" in p for p in g.validate_diagram_data(leaky)))

    ragged = {
        "metadata": {"sub_topic": "Line Graph DI"},
        "text": "read the chart",
        "diagram_data": {
            "chart_type": "line",
            "x_labels": ["Jan", "Feb", "Mar"],
            "series": [{"name": "P", "values": [1, 2]}],   # 2 values vs 3 labels
        },
    }
    check("ragged series vs x_labels flagged",
          any("align" in p for p in g.validate_diagram_data(ragged)))

    arch = {
        "metadata": {"sub_topic": "VPC Peering", "topic": "Networking"},
        "text": "A company has six VPCs across two regions...",
        "diagram_data": None,
    }
    check("architecture (non-chart) is never flagged",
          g.validate_diagram_data(arch) == [])
    check("is_chart_question false for architecture", not g.is_chart_question(arch))


# ── 3. Score parsing ─────────────────────────────────────────────────────────
def test_score_parsing():
    print("\n── Diagram-score parsing ──")
    check("SCORE: 5 -> 5", g.parse_diagram_score("SCORE: 5\nlooks great")[0] == 5)
    check("SCORE: 2 -> 2", g.parse_diagram_score("SCORE: 2\n1. D2 issue")[0] == 2)
    check("bare PASS -> 5", g.parse_diagram_score("PASS")[0] == 5)
    check("no score sig -> conservative (<pass)",
          g.parse_diagram_score("1. something is wrong")[0] < g.DIAGRAM_PASS_SCORE)


# ── 4. Live renderer round-trip (optional) ───────────────────────────────────
def test_renderer():
    print(f"\n── Renderer round-trip ({g.RENDERER_URL}) ──")
    code = (r"\documentclass[border=4mm]{standalone}\usepackage{tikz}"
            r"\usetikzlibrary{arrows.meta}"
            r"\begin{document}\begin{tikzpicture}"
            r"\node[draw,rounded corners] (a) at (0,0){Hello};"
            r"\node[draw,rounded corners] (b) at (3,0){World};"
            r"\draw[-{Stealth}] (a)--(b);"
            r"\end{tikzpicture}\end{document}")
    res = g.render_tikz(code)

    if not res["ok"] and res["error"] and "unreachable" in str(res["error"]).lower():
        print("  ⏭️  SKIPPED — renderer not running "
              "(start it with `npm run dev -- -p 3002`, or set RENDERER_URL)")
        return

    check("render ok", res["ok"], str(res.get("error")))
    check("svg returned", bool(res.get("svg")))
    check("png_base64 returned", bool(res.get("png_base64")))
    check("clean diagram has 0 overfull boxes", res.get("overfull", 0) == 0,
          f"overfull={res.get('overfull')}")

    if res.get("png_base64"):
        out = "/tmp/test_pipeline_render.png"
        with open(out, "wb") as f:
            f.write(base64.b64decode(res["png_base64"]))
        print(f"     (wrote {out} — open it to eyeball the render)")

    # A narrow fixed-width box with an unbreakable long word forces an overfull \hbox
    # (standalone's page auto-grows, so we must constrain a box explicitly to trip it).
    overfull = (r"\documentclass[border=4mm]{standalone}\usepackage{tikz}"
                r"\begin{document}\begin{tikzpicture}"
                r"\node[draw,text width=1cm] at (0,0){" + ("W" * 80) + r"};"
                r"\end{tikzpicture}\end{document}")
    of_res = g.render_tikz(overfull)
    if of_res["ok"]:
        check("constrained over-long content reports overfull boxes",
              of_res.get("overfull", 0) > 0,
              f"overfull={of_res.get('overfull')} (expected > 0)")
    else:
        print("  ⏭️  overfull check skipped (it failed to compile, which is also acceptable)")


def test_math_verifier():
    print("\n── Math verifier (sympy) ──")
    if not getattr(g, "_SYMPY_AVAILABLE", False):
        print("  ⏭️  SKIPPED — sympy not installed (pip install sympy)")
        return

    # Clean: (22*3 + 100 - 19)/7 = 21 = option C
    clean = {"options": {"A": "18", "B": "19", "C": "21", "D": "22"}, "correct_answer": "C",
             "computation": {"expression": "(22*3 + 100 - 19)/7", "computed_value": 21}}
    check("consistent numeric question passes", g.verify_computation(clean) == [],
          str(g.verify_computation(clean)))

    # Broken (real log case): expression = 17.35, options 18/19/21/22
    broken = {"options": {"A": "18", "B": "19", "C": "21", "D": "22"}, "correct_answer": "C",
              "computation": {"expression": "(sqrt(676)*cbrt(216)+14^2-root(625,4))/"
                                            "(sqrt(289)-cbrt(125)+4^3/8)", "computed_value": 21}}
    check("expression matching no option is flagged",
          any("matches NONE" in p for p in g.verify_computation(broken)))

    # computed_value disagrees with the expression
    mismatch = {"options": {"A": "813", "B": "962"}, "correct_answer": "B",
                "computation": {"expression": "800 + 13", "computed_value": 962}}
    check("computed_value vs expression mismatch flagged",
          any("≠ Python evaluation" in p for p in g.verify_computation(mismatch)))

    # value matches B but correct_answer says C
    wrongkey = {"options": {"A": "18", "B": "21", "C": "19"}, "correct_answer": "C",
                "computation": {"expression": "147/7", "computed_value": 21}}
    check("wrong answer-key flagged with correction",
          any("Set correct_answer to 'B'" in p for p in g.verify_computation(wrongkey)))

    # approximation within tolerance passes
    approx = {"options": {"A": "3.9", "B": "4.5", "C": "5.8"}, "correct_answer": "B",
              "computation": {"expression": "(4.98**2*sqrt(224.8))/(2.03**3+11.97*6.05)",
                              "computed_value": 4.6, "tolerance": 0.3}}
    check("approximation within tolerance passes", g.verify_computation(approx) == [],
          str(g.verify_computation(approx)))

    # no computation field → not checked (e.g. conceptual/verbal question)
    conceptual = {"options": {"A": "x"}, "correct_answer": "A"}
    check("non-numeric question is skipped", g.verify_computation(conceptual) == [])

    # stem ↔ computation drift (real log case): stem √1444=62, computation √1296=60
    drift = {"text": r"Simplify: $\sqrt{1444} + 14 - 6 + 16$",
             "options": {"A": "56", "B": "58", "C": "60", "D": "62"}, "correct_answer": "C",
             "computation": {"expression": "sqrt(1296) + 14 - 6 + 16", "computed_value": 60}}
    check("stem↔computation drift is flagged",
          any("STEM ↔ COMPUTATION" in p for p in g.verify_computation(drift)))

    # stem matches computation → passes
    aligned = {"text": r"Simplify: $\sqrt{1296} + 14 - 6 + 16$",
               "options": {"A": "56", "B": "58", "C": "60", "D": "62"}, "correct_answer": "C",
               "computation": {"expression": "sqrt(1296) + 14 - 6 + 16", "computed_value": 60}}
    check("stem matching computation passes", g.verify_computation(aligned) == [],
          str(g.verify_computation(aligned)))

    # word problem (no extractable stem expression) → stem check skipped, no false positive
    word = {"text": "A train travels 360 km in 4 hours. What is its average speed in km/h?",
            "options": {"A": "80", "B": "90", "C": "100", "D": "75"}, "correct_answer": "B",
            "computation": {"expression": "360/4", "computed_value": 90}}
    check("word problem (no stem math) has no false positive", g.verify_computation(word) == [],
          str(g.verify_computation(word)))

    # ── Authoritative mode (apply_fixes=True): Python owns answer + options ──
    import copy

    # matches NONE → Python rebuilds options so the value is present; no reject
    nomatch = {"text": r"$5^2 + 14$", "options": {"A": "18", "B": "19", "C": "21", "D": "22"},
               "correct_answer": "C", "computation": {"expression": "5**2 + 14", "computed_value": 99}}
    q = copy.deepcopy(nomatch); fixes = []
    probs = g.verify_computation(q, apply_fixes=True, fixes_out=fixes)
    ov = g._option_to_float(q["options"][q["correct_answer"]])
    check("authoritative: no-match rebuilds options (no reject)",
          probs == [] and abs(ov - 39) < 0.05 and len(q["options"]) == 4,
          f"probs={probs} opt={ov} n={len(q['options'])}")

    # options already valid → kept, only key corrected
    keyfix = {"text": r"$147/7$", "options": {"A": "18", "B": "21", "C": "19", "D": "25"},
              "correct_answer": "A", "computation": {"expression": "147/7", "computed_value": 21}}
    q = copy.deepcopy(keyfix); fixes = []
    g.verify_computation(q, apply_fixes=True, fixes_out=fixes)
    check("authoritative: valid options kept, key fixed to B",
          q["correct_answer"] == "B" and q["options"] == keyfix["options"], str(q["options"]))

    # stem drift → still hard-rejected even in authoritative mode (no rebuild)
    q = copy.deepcopy(drift); fixes = []
    probs = g.verify_computation(q, apply_fixes=True, fixes_out=fixes)
    check("authoritative: stem drift still rejected (not auto-answered)",
          any("STEM ↔ COMPUTATION" in p for p in probs) and not fixes)

    # ── Robustness fixes (from real-run bugs) ──
    # stem parser must NOT return a false 0 on nested \frac{...\div...}{\sqrt{}-n}
    fracstem = r"Evaluate $\frac{4356 \div 66 + 18 \times 5}{\sqrt{225} - 2}$"
    check("nested frac stem parses correctly (not false 0)",
          abs((g._extract_stem_value(fracstem) or -1) - 12.0) < 0.01,
          str(g._extract_stem_value(fracstem)))

    # unparseable/garbage stem → None (skip), never a wrong number
    check("garbage latex → None (skip, no false value)",
          g._safe_eval_expr(r"\foo{3}{4}") is None and g._safe_eval_expr("3 + x") is None)

    # option that is a fraction must be EVALUATED, not read as its numerator
    check("fraction option evaluated to true value",
          abs((g._option_to_float(r"$\frac{2.5789}{2}$") or 0) - 1.28945) < 0.01,
          str(g._option_to_float(r"$\frac{2.5789}{2}$")))

    # broken-template options force a rebuild (true value can't hide behind a fraction)
    broken = {"text": r"$98/38$",
              "options": {"A": r"$\frac{2.4}{2}$", "B": r"$\frac{2.5789}{2}$",
                          "C": r"$\frac{2.7}{2}$", "D": r"$\frac{2.9}{2}$"},
              "correct_answer": "B",
              "computation": {"expression": "98/38", "computed_value": 2.5789,
                              "tolerance": 0.01, "answer_option": "A"}}
    q = copy.deepcopy(broken); fixes = []
    g.verify_computation(q, apply_fixes=True, fixes_out=fixes)
    ov = g._option_to_float(q["options"][q["correct_answer"]])
    check("broken fraction-options rebuilt to clean values",
          abs((ov or 0) - 2.5789) < 0.01 and q["computation"]["answer_option"] == q["correct_answer"],
          f"opt={ov} ao={q['computation']['answer_option']} key={q['correct_answer']}")


def test_stem_from_computation():
    print("\n── Stem-from-computation (expr → LaTeX) ──")
    check("renders sqrt/cbrt/frac in order",
          g.expr_to_latex("sqrt(1296) + cbrt(216) - sqrt(441) + 4**3/8")
          == r"\sqrt{1296} + \sqrt[3]{216} - \sqrt{441} + \frac{4^{3}}{8}",
          g.expr_to_latex("sqrt(1296) + cbrt(216) - sqrt(441) + 4**3/8"))
    check("x**0.5 becomes \\sqrt",
          r"\sqrt{529}" in (g.expr_to_latex("529**0.5 + 5") or ""),
          g.expr_to_latex("529**0.5 + 5"))
    check("fraction-of-fraction stem renders",
          g.expr_to_latex("(7**3 + 5**3) / (7**2 - 7*5 + 5**2)")
          == r"\frac{7^{3} + 5^{3}}{7^{2} - 7 \times 5 + 5^{2}}",
          g.expr_to_latex("(7**3 + 5**3) / (7**2 - 7*5 + 5**2)"))
    check("garbage expression → None", g.expr_to_latex("foo bar }{") is None)

    # End-to-end: placeholder → rendered stem → verifies by construction
    q = {"text": "Simplify the following expression: [[EXPR]]",
         "options": {"A": "50", "B": "51", "C": "52", "D": "53"}, "correct_answer": "B",
         "computation": {"expression": "529**0.5 + 343**(1/3) - 1024**0.5/8 + 5**2",
                         "computed_value": 51}}
    rendered = g.expr_to_latex(q["computation"]["expression"])
    q["text"] = q["text"].replace(g._STEM_MATH_PLACEHOLDER, f"${rendered}$")
    check("placeholder produces a self-consistent stem",
          g._STEM_MATH_PLACEHOLDER not in q["text"]
          and g.verify_computation(g_copy(q), apply_fixes=True) == [],
          q["text"])


def test_structure_fingerprint():
    print("\n── Question variety (structure fingerprint) ──")
    q1 = {"text": r"Simplify: $\sqrt{529} + \sqrt{361} - \sqrt{256} + 3^3$",
          "computation": {"expression": "23+19-16+3**3"}}
    q2 = {"text": r"Simplify: $\sqrt{625} + \sqrt{169} - \sqrt{81} + 7^2$",
          "computation": {"expression": "25+13-9+7**2"}}
    check("same template, different numbers → same fingerprint",
          g.structure_fingerprint(q1) == g.structure_fingerprint(q2),
          f"{g.structure_fingerprint(q1)} vs {g.structure_fingerprint(q2)}")

    word = {"text": "A train covers 360 km in 4 hours. Find its average speed.",
            "computation": {"expression": "360/4"}}
    check("word problem → different fingerprint than surd simplification",
          g.structure_fingerprint(word) != g.structure_fingerprint(q1),
          g.structure_fingerprint(word))
    check("'Simplify:' colon does not falsely trigger 'ratio'",
          "ratio" not in g.structure_fingerprint(q1), g.structure_fingerprint(q1))


def g_copy(obj):
    import copy
    return copy.deepcopy(obj)


def main():
    print("=" * 60)
    print("Diagram pipeline offline tests (no AWS)")
    print("=" * 60)
    test_lint()
    test_chart_validation()
    test_score_parsing()
    test_math_verifier()
    test_stem_from_computation()
    test_structure_fingerprint()
    test_renderer()

    print("\n" + "=" * 60)
    print(f"RESULT: {_PASS} passed, {_FAIL} failed")
    print("=" * 60)
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
