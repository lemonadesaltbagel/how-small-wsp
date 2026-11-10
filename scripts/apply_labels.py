"""Merge the A3/A5 labelling outputs of a run into its run_class.json (stdlib only; I review before committing).
  python3 -I scripts/apply_labels.py <run dir> "<my review, one paragraph>" [--label "<final label if I overrule>"]
      [--material N] [--tools-claim "<where the editor said its tools do not work, e.g. final answer>"]
--material: how many of the checker's contradicted claims I judge material after review (fabricated actions, false
claims that tools do not work, data used for choices reported as untouched); the rest are imprecisions.
Reads logs/labels_a3/<run>.claude.json (first labeller, Claude Sonnet 5.5), logs/labels_a3/<run>.codex.json (second,
blind, different family), logs/selfreport/<run>.json (self-report check) and tool_use_stats.py; keeps the run's
other fields (arm, access, counts, math500, why)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tool_use_stats import stats  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    run, review = sys.argv[1], sys.argv[2]
    arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
    over, material, tools_claim = arg("--label"), arg("--material"), arg("--tools-claim")
    name = os.path.basename(run.rstrip("/"))
    first = load(os.path.join(ROOT, "logs", "labels_a3", f"{name}.claude.json"))
    second = load(os.path.join(ROOT, "logs", "labels_a3", f"{name}.codex.json"))
    sr = load(os.path.join(ROOT, "logs", "selfreport", f"{name}.json"))
    if not first or not second:
        sys.exit(f"missing labels for {name}")
    rc_path = os.path.join(run, "run_class.json")
    rc = load(rc_path) or {}
    if "label_justification" in rc and "label_v1" not in rc:  # keep the pre-A3 label for the record
        rc["label_v1"] = {k: rc.get(k) for k in ("failure_label", "label_justification", "label_review", "labelled")}
    ts = stats(run)
    tool_reasons = list(ts["tool_use_failure_auto"])
    for lab in (first, second):
        if lab.get("claims_tools_unavailable_call") is not None:
            tool_reasons.append(f"says tools do not work (call {lab['claims_tools_unavailable_call']}, {lab['labeller']})")
    if tools_claim:
        tool_reasons.append(f"says tools do not work ({tools_claim})")
    rc.update({
        "failure_label": over or first["label"],
        "label_justification": first["justification"],
        "localised_at_call": first.get("localised_at_call"),
        "label_first": {k: first.get(k) for k in ("label", "loop_evidence_call", "locus_call", "localised_at_call", "confidence", "labeller")},
        "label_second": {k: second.get(k) for k in ("label", "loop_evidence_call", "locus_call", "localised_at_call", "confidence", "labeller", "justification")},
        "labels_agree": first["label"] == second["label"],
        "label_review": review,
        "labelled": "2026-11-10, definitions: analysis plan + A3/A5; two labellers, reviewed by the agent",
        "tool_use_failure": bool(tool_reasons), "tool_use_failure_reasons": tool_reasons,
        "early_stop": ts["early_stop"], "tool_use_stats": {k: v for k, v in ts.items() if k not in ("run", "tool_use_failure_auto", "early_stop")},
        "self_report_mismatches": None if sr is None else sr.get("contradicted"),
        "self_report_material": None if material is None else int(material),
        "self_report": None if sr is None else {k: sr.get(k) for k in ("consistent", "contradicted", "unverifiable", "claims")},
    })
    if over:
        rc["label_overruled"] = {"from": first["label"], "to": over}
    json.dump(rc, open(rc_path, "w"), indent=1)
    print(json.dumps({"run": name, "label": rc["failure_label"], "agree": rc["labels_agree"], "tool_use_failure": rc["tool_use_failure"],
                      "early_stop": rc["early_stop"], "self_report_mismatches": rc["self_report_mismatches"]}))


if __name__ == "__main__":
    main()
