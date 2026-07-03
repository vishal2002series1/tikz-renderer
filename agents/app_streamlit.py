"""
app_streamlit.py — GUI for the exam question-bank generator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A point-and-click front end over generic_exam_question_bank.run_seeder.

Features
  • Cascading scope selectors (Exam → Subject → Topic → Sub-topic), all multi-select.
  • All tuning knobs exposed (questions/level, iterations, difficulty levels, diagram mode,
    retry budgets, output file).
  • One-click Generate, with live log streaming and a progress bar.
  • Each banked question shown as a card — text, options (correct highlighted), explanation,
    and the rendered figure (PNG) when present.
  • Download the resulting JSON bank.
  • Renderer health check so you know the TikZ service is up before generating.

Run it:
    cd agents
    ./venv/bin/python -m streamlit run app_streamlit.py
    # (make sure the TikZ renderer is running:  npm run dev -- -p 3002  in the project root)
"""

import os
import io
import json
import base64
import queue
import threading
import contextlib

import requests
import streamlit as st

import generic_exam_question_bank as core

st.set_page_config(page_title="Exam Question Bank Generator", page_icon="🧠", layout="wide")

SYLLABUS_FILE = "syllabus_maps.json"


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_syllabus(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def renderer_ok() -> tuple:
    """Ping the renderer with a trivial diagram. Returns (ok, message)."""
    code = (r"\documentclass[border=2mm]{standalone}\usepackage{tikz}"
            r"\begin{document}\begin{tikzpicture}\node{ok};\end{tikzpicture}\end{document}")
    try:
        res = core.render_tikz(code)
        if res.get("ok"):
            return True, "Renderer reachable ✓"
        return False, f"Renderer error: {res.get('error')}"
    except Exception as e:
        return False, f"Renderer unreachable: {e}"


def svg_to_png_bytes(svg_path: str):
    """Re-render a banked question's TikZ to PNG via the renderer, for display."""
    return None  # placeholder; we render from TikZ_Code instead (see render_question_png)


def render_question_png(q: dict):
    """Render a question's TikZ_Code to PNG bytes via the renderer (or None)."""
    code = q.get("TikZ_Code")
    if not code:
        return None
    try:
        res = core.render_tikz(code)
        b64 = res.get("png_base64")
        if b64:
            return base64.b64decode(b64)
    except Exception:
        pass
    return None


def as_selector(values, all_label="All"):
    """Convert a multiselect result into the core's selector form ('All' or a list)."""
    if not values or all_label in values:
        return "All"
    return list(values)


# ──────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False
if "log_lines" not in st.session_state:
    st.session_state.log_lines = []
if "banked" not in st.session_state:
    st.session_state.banked = []
if "worker" not in st.session_state:
    st.session_state.worker = None
if "msg_queue" not in st.session_state:
    st.session_state.msg_queue = None
if "stop_flag" not in st.session_state:
    st.session_state.stop_flag = None
if "progress" not in st.session_state:
    st.session_state.progress = (0, 0)


# ──────────────────────────────────────────────────────────────────────────
# Sidebar — configuration
# ──────────────────────────────────────────────────────────────────────────
syllabus = load_syllabus(SYLLABUS_FILE)

st.sidebar.title("⚙️  Configuration")

exam = st.sidebar.selectbox("Exam", list(syllabus.keys()))
exam_data = core.unwrap_exam_data(syllabus, exam)
scope = core.list_scope(exam_data)

# Subject (multi-select; empty == All)
subjects_sel = st.sidebar.multiselect(
    "Subject(s)", scope["subjects"], default=[],
    help="Leave empty for ALL subjects.")
chosen_subjects = subjects_sel or scope["subjects"]

# Topic (only if exam is nested) — union of topics across chosen subjects
topic_sel = []
if not scope["flat"]:
    topic_options = []
    for s in chosen_subjects:
        topic_options += scope["topics_by_subject"].get(s, [])
    topic_options = sorted(set(topic_options))
    topic_sel = st.sidebar.multiselect(
        "Topic(s)", topic_options, default=[],
        help="Leave empty for ALL topics under the chosen subject(s).")
chosen_topics = topic_sel or (
    sorted({t for s in chosen_subjects for t in scope["topics_by_subject"].get(s, [])})
    if not scope["flat"] else [""])

# Sub-topic — union across chosen (subject, topic) pairs
subtopic_options = []
for (s, t), sts in scope["subtopics"].items():
    if s in chosen_subjects and (scope["flat"] or t in chosen_topics):
        subtopic_options += sts
subtopic_options = sorted(set(subtopic_options))
subtopic_sel = st.sidebar.multiselect(
    "Sub-topic(s)", subtopic_options, default=[],
    help="Leave empty for ALL sub-topics in scope.")

st.sidebar.markdown("---")

# Generation knobs
n_per_level = st.sidebar.number_input("Questions per difficulty level", 1, 50, 2)
k_iterations = st.sidebar.number_input("Iterations (K)", 1, 10, 1,
                                       help="K=2 doubles the total questions.")
difficulty_levels = st.sidebar.multiselect(
    "Difficulty levels", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])

diagram_mode = st.sidebar.radio(
    "Diagram mode", ["auto", "force", "never"], index=0,
    help="auto = model decides · force = every Q has a figure · never = text-only.")

with st.sidebar.expander("Advanced"):
    max_rounds = st.number_input(
        "Max rounds per slot (0 = unlimited)", 0, 50, 5,
        help="Give up on a question slot after this many full retry rounds.")
    default_out = (f"{exam}_{'_'.join(subjects_sel) or 'All'}"
                   .replace(' ', '_').replace('/', '-')[:60] + ".json")
    output_file = st.text_input("Output JSON file", value=default_out)

st.sidebar.markdown("---")
ok, rmsg = renderer_ok()
(st.sidebar.success if ok else st.sidebar.error)(rmsg)
st.sidebar.caption(f"Renderer: {core.RENDERER_URL}")


# ──────────────────────────────────────────────────────────────────────────
# Worker thread
# ──────────────────────────────────────────────────────────────────────────
class _QueueWriter(io.TextIOBase):
    """Redirects print() output into a thread-safe queue for live log streaming."""
    def __init__(self, q): self.q = q
    def write(self, s):
        if s and s.strip():
            self.q.put(("log", s.rstrip("\n")))
        return len(s)


def run_generation(cfg, msg_queue, stop_event):
    def on_question(q_data, count):
        msg_queue.put(("question", q_data))

    def on_progress(done, total, label):
        msg_queue.put(("progress", (done, total, label)))

    writer = _QueueWriter(msg_queue)
    try:
        with contextlib.redirect_stdout(writer):
            core.run_seeder(
                exam=cfg["exam"],
                subject=cfg["subject"],
                topic=cfg["topic"],
                sub_topic=cfg["sub_topic"],
                n_per_level=cfg["n_per_level"],
                k_iterations=cfg["k_iterations"],
                difficulty_levels=cfg["difficulty_levels"],
                syllabus_file=SYLLABUS_FILE,
                output_file=cfg["output_file"],
                diagram_mode=cfg["diagram_mode"],
                on_question=on_question,
                on_progress=on_progress,
                max_rounds_per_slot=cfg["max_rounds"],
                should_stop=stop_event.is_set,
            )
    except Exception as e:
        msg_queue.put(("log", f"❌ Run crashed: {e}"))
    finally:
        msg_queue.put(("done", None))


# ──────────────────────────────────────────────────────────────────────────
# Main panel
# ──────────────────────────────────────────────────────────────────────────
st.title("🧠 Exam Question Bank Generator")
st.caption("Generate exam-accurate MCQs with auto-verified math and rendered TikZ figures.")

# Scope summary
n_subtopics = len([1 for (s, t), sts in scope["subtopics"].items()
                   if s in chosen_subjects and (scope["flat"] or t in chosen_topics)
                   for _ in sts])
sel_subtopics = subtopic_sel or None
est_subtopics = len(subtopic_sel) if subtopic_sel else n_subtopics
est_total = est_subtopics * max(len(difficulty_levels), 0) * k_iterations * n_per_level

c1, c2, c3 = st.columns(3)
c1.metric("Sub-topics in scope", est_subtopics)
c2.metric("Difficulty levels", len(difficulty_levels))
c3.metric("Est. questions", est_total)

col_run, col_stop = st.columns([1, 1])
start = col_run.button("▶️  Generate", type="primary", disabled=st.session_state.running,
                       use_container_width=True)
stop = col_stop.button("⏹  Stop", disabled=not st.session_state.running,
                       use_container_width=True)

if start:
    if not ok:
        st.error("Renderer is not reachable. Start it with `npm run dev -- -p 3002` "
                 "(or set RENDERER_URL) before generating diagrams.")
    elif not difficulty_levels:
        st.error("Pick at least one difficulty level.")
    else:
        st.session_state.running = True
        st.session_state.log_lines = []
        st.session_state.banked = []
        st.session_state.progress = (0, est_total)
        q = queue.Queue()
        stop_event = threading.Event()
        st.session_state.msg_queue = q
        st.session_state.stop_flag = stop_event
        cfg = {
            "exam": exam,
            "subject": as_selector(subjects_sel),
            "topic": "All" if scope["flat"] else as_selector(topic_sel),
            "sub_topic": as_selector(subtopic_sel),
            "n_per_level": int(n_per_level),
            "k_iterations": int(k_iterations),
            "difficulty_levels": sorted(difficulty_levels),
            "diagram_mode": diagram_mode,
            "output_file": output_file,
            "max_rounds": int(max_rounds),
        }
        t = threading.Thread(target=run_generation, args=(cfg, q, stop_event), daemon=True)
        t.start()
        st.session_state.worker = t
        st.rerun()

if stop and st.session_state.stop_flag is not None:
    st.session_state.stop_flag.set()
    st.toast("Stop requested — finishing the current question…")

# Drain the queue
if st.session_state.msg_queue is not None:
    q = st.session_state.msg_queue
    drained = False
    while True:
        try:
            kind, payload = q.get_nowait()
        except queue.Empty:
            break
        drained = True
        if kind == "log":
            st.session_state.log_lines.append(payload)
        elif kind == "question":
            st.session_state.banked.append(payload)
        elif kind == "progress":
            done, total, _label = payload
            st.session_state.progress = (done, total)
        elif kind == "done":
            st.session_state.running = False

# Progress bar
done, total = st.session_state.progress
if total:
    st.progress(min(done / total, 1.0),
                text=f"{done}/{total} slots · {len(st.session_state.banked)} banked")

# Layout: questions (left) + live log (right)
left, right = st.columns([3, 2])

with right:
    st.subheader("📜 Live log")
    st.code("\n".join(st.session_state.log_lines[-200:]) or "(idle)", language="text")

with left:
    st.subheader(f"📦 Banked questions ({len(st.session_state.banked)})")
    if st.session_state.banked:
        data = json.dumps(st.session_state.banked, indent=2, ensure_ascii=False)
        st.download_button("⬇️  Download JSON", data,
                           file_name=output_file, mime="application/json")
    for q_data in reversed(st.session_state.banked[-30:]):
        icon = "📐" if q_data.get("Requires_Diagram") else "📝"
        meta = q_data.get("metadata", {})
        with st.expander(f"{icon}  {q_data.get('id','?')}  —  "
                         f"L{meta.get('difficulty_level','?')} · {meta.get('sub_topic','')}"):
            st.markdown(q_data.get("text", ""))
            correct = q_data.get("correct_answer")
            for key, val in (q_data.get("options") or {}).items():
                mark = "✅ " if key == correct else "   "
                st.markdown(f"{mark}**{key}.** {val}")
            if q_data.get("Requires_Diagram"):
                png = render_question_png(q_data)
                if png:
                    st.image(png, caption="Rendered figure")
                else:
                    st.caption("⚠️ Figure could not be rendered for preview.")
            with st.expander("Explanation"):
                st.markdown(q_data.get("explanation", ""))

# Auto-refresh while running so the queue keeps draining
if st.session_state.running:
    import time
    time.sleep(1.0)
    st.rerun()
