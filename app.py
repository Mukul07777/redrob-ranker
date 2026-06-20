import gradio as gr
import json
import sys
sys.path.insert(0, ".")
from rank import score_candidate, generate_reasoning, is_honeypot


def rank_candidates(jsonl_text):
    if not jsonl_text.strip():
        return "Paste at least one candidate JSON line above."

    candidates = []
    for i, line in enumerate(jsonl_text.strip().split("\n")):
        line = line.strip()
        if not line:
            continue
        try:
            candidates.append(json.loads(line))
        except json.JSONDecodeError as e:
            return f"Error on line {i+1}: {e}"

    if not candidates:
        return "No valid candidates found."

    results = []
    for c in candidates:
        cid = c.get("candidate_id", "UNKNOWN")
        if is_honeypot(c):
            score = 0.001
            flag = " HONEYPOT"
        else:
            score = score_candidate(c)
            flag = ""
        results.append((cid, score, c, flag))

    results.sort(key=lambda x: (-x[1], x[0]))

    output = []
    for rank, (cid, score, c, flag) in enumerate(results, 1):
        reasoning = generate_reasoning(c, rank)
        output.append(f"#{rank} | {cid} | Score: {score:.4f}{flag}")
        output.append(f"     {reasoning}")
        output.append("")

    return "\n".join(output)


with gr.Blocks() as demo:
    gr.Markdown("# Redrob Candidate Ranker")
    gr.Markdown("Paste candidate records (one per line as JSON) to rank them against the Senior AI Engineer JD.")
    inp = gr.Textbox(lines=20, label="Candidate JSONL (one JSON per line)")
    btn = gr.Button("Rank Candidates")
    out = gr.Textbox(lines=30, label="Ranked Results")
    btn.click(fn=rank_candidates, inputs=inp, outputs=out)

demo.launch(server_name="0.0.0.0", server_port=7860)