"""Figure 1 of the preprint from the episode ledger (no numbers typed by hand except the fixed reference values below,
each with its source). Run on a node with matplotlib (desktop sandbox venv):
  harness/assets/venvs/sandbox/bin/python -I scripts/make_figures.py   -> paper/figs/fig1_scale_curve.{pdf,png}
Only runs whose run_class.json says counts=True and whose task is strict2 enter the curve (analysis plan)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LEDGER = os.path.join(ROOT, "notes", "episode_ledger.json")
OUT = os.path.join(ROOT, "paper", "figs")

# Fixed references (sources: notes/episode_ledger.md, notes/phase0_log.md, the authors' logs)
BASE = 0.090                 # unedited Qwen3.5-0.8B, MATH-500, 1 sample (ours)
REPAIR = BASE + 0.20         # repair threshold (analysis plan)
OPUS_STRICT = [0.448, 0.554, 0.406]
ASTRA = 0.481                # authors' p50c, 4 samples; its log has 3 rule-(b) linear attributions
FLOOR = 0.336                # KidIkaros scripted edit, given the paper's localisation (1 sample)
THINK_OFF = 0.4895           # unedited base with thinking turned off, 4 samples (notes/pending_user_decisions.md 2026-11-13 18:40)
SIZES = {"qwen0p8b": 0.8, "qwen9b": 9.0, "qwen27b": 27.0}
DS_INTERVAL = (16.0, 748.0)  # deepseek-flash: activated (decode) .. total (backbone + Engram), model card

# Categorical slots 1-2 of the reference palette, validated (scripts/validate_palette.js, light: all checks pass)
COLOR = {"hinted": "#2a78d6", "unhinted": "#eb6834"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def editor_and_prompt(run):
    name = os.path.basename(run)
    editor = name.split("-")[0]
    prompt = "unhinted" if "-unhinted-" in name else "hinted"
    return editor, prompt


def main():
    rows = json.load(open(LEDGER))
    pts = {}  # (editor, prompt) -> [scores]
    for r in rows:
        if not r.get("counts") or "strict2" not in r["run"] or r.get("math500") is None:
            continue
        pts.setdefault(editor_and_prompt(r["run"]), []).append((r["math500"], bool(r.get("empty_edit"))))

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 1500)
    ax.set_ylim(0, 0.62)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Editor parameters (billions, log scale)", color=INK, fontsize=9)
    ax.set_ylabel("MATH-500 of the submitted edit", color=INK, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xticks([0.8, 9, 27, 100, 748])
    ax.set_xticklabels(["0.8", "9", "27", "100", "748"])

    # reference lines (neutral ink, direct labels; not series colors)
    # (y, label, linestyle, ink, label below the line?)
    refs = [(FLOOR, "scripted floor (given localisation)", ":", MUTED, False),
            (REPAIR, "repair threshold (+20 pts)", (0, (1, 3)), MUTED, False), (BASE, "unedited base", "-", MUTED, False)]
    for y, label, ls, ink, below in refs:
        ax.axhline(y, color=ink, linestyle=ls, linewidth=1.0, zorder=1)
        ax.text(1450, y - 0.006 if below else y + 0.006, label, color=ink, fontsize=7, ha="right",
                va="top" if below else "bottom")
    # Opus, Astra and thinking off lie within 2 points of each other: a small key instead of direct labels
    close = [(sum(OPUS_STRICT) / 3, "Opus 5.5, strict (mean of 3)", "-", INK),
             (ASTRA, "Astra (log: 3 rule-(b) violations)", "--", MUTED),
             (THINK_OFF, "unedited, thinking off", "-.", INK)]
    handles = [ax.axhline(y, color=ink, linestyle=ls, linewidth=1.0, zorder=1, label=label) for y, label, ls, ink in close]
    key = ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.58), fontsize=7, frameon=False)
    for t in key.get_texts():
        t.set_color(INK)
    ax.add_artist(key)

    for prompt, c in COLOR.items():
        xs, ys = [], []
        dx = 0.94 if prompt == "hinted" else 1.06  # small offset on the log axis so the two prompts do not overlap
        for ed, size0 in sorted(SIZES.items(), key=lambda kv: kv[1]):
            size = size0 * dx
            for y, empty in pts.get((ed, prompt), []):
                # amendment A4: an empty edit set is drawn hollow at its measured score
                ax.plot(size, y, "o", color=c, markersize=5, zorder=3, markerfacecolor="white" if empty else c,
                        markeredgecolor=c if empty else "white", markeredgewidth=1.2 if empty else 1.0)
            if pts.get((ed, prompt)):
                ys_cell = [y for y, _ in pts[(ed, prompt)]]
                xs.append(size); ys.append(sum(ys_cell) / len(ys_cell))
        if xs:
            ax.plot(xs, ys, "-", color=c, linewidth=2, zorder=2, label=f"Qwen3.5, {prompt} prompt")
        for y, empty in pts.get(("deepseekflash", prompt), []):
            ax.plot(DS_INTERVAL, (y, y), "-", color=c, linewidth=2, alpha=0.6, zorder=2)
            ax.plot(DS_INTERVAL, (y, y), "D", color=c, markersize=5, zorder=3, markerfacecolor="white" if empty else c,
                    markeredgecolor=c if empty else "white", markeredgewidth=1.2 if empty else 1.0)
    ax.plot([], [], "D-", color=MUTED, markersize=5, linewidth=2, label="deepseek-flash (activated → total)")
    ax.plot([], [], "o", color=MUTED, markersize=5, markerfacecolor="white", linestyle="none", label="empty edit set (unedited model re-sampled)")
    series = [h for h in ax.get_lines() if h not in handles and not h.get_label().startswith("_")]
    leg = ax.legend(handles=series, loc="lower right", bbox_to_anchor=(1.0, 0.19), fontsize=7, frameon=False)
    for t in leg.get_texts():
        t.set_color(INK)
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig1_scale_curve.{ext}"))
    print("points:", {f"{k[0]}/{k[1]}": v for k, v in pts.items()})  # (score, empty edit)


if __name__ == "__main__":
    main()
