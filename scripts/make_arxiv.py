"""Build the arXiv upload from paper/preprint.md: paper/arxiv/ (main.tex + figs/*.pdf) and paper/how-small-wsp-arxiv.zip,
then compile a check PDF to paper/how-small-wsp.pdf. Needs pandoc and tectonic (Mac); arXiv compiles main.tex with
pdflatex, so main.tex is made pure ASCII (every non-ASCII character becomes a standard LaTeX command) and uses only
the packages of pandoc's default template.
  python3 -I scripts/make_arxiv.py"""
import os
import re
import shutil
import subprocess
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
OUT = os.path.join(PAPER, "arxiv")
ZIP = os.path.join(PAPER, "how-small-wsp-arxiv.zip")
PDF = os.path.join(PAPER, "how-small-wsp.pdf")
FIGS = ["fig1_scale_curve.pdf", "fig2_failure_by_size.pdf"]
AUTHOR = "Xintian Huang"
# pandoc extensions off: our text has literal $ amounts, a lone ~, </think> outside code and no raw TeX
FROM = "markdown+lists_without_preceding_blankline-tex_math_dollars-subscript-superscript-raw_tex-raw_html-implicit_figures"
HEADER = r"""\usepackage{listings}
\lstset{basicstyle=\ttfamily\scriptsize,breaklines=true,columns=fullflexible,keepspaces=true,upquote=true}
\usepackage{textcomp}
\setlength{\emergencystretch}{3em}
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\footnotesize}
\setlength{\tabcolsep}{3pt}
"""
ASCII = {
    "d̂": r"\ensuremath{\hat{d}}", "–": "--", "—": "---", "−": r"\ensuremath{-}", "§": r"\S{}",
    "¥": r"\textyen{}", "·": r"\ensuremath{\cdot}", "±": r"\ensuremath{\pm}",
    "→": r"\ensuremath{\rightarrow}", "×": r"\ensuremath{\times}", "δ": r"\ensuremath{\delta}",
    "β": r"\ensuremath{\beta}", "…": r"\ldots{}", "≥": r"\ensuremath{\geq}",
    "ᵀ": r"\ensuremath{^{\mathsf{T}}}", "‘": "`", "’": "'", "“": "``", "”": "''",
}


def split_front(md):
    """Title, abstract and body; the draft's author/date line is replaced by the title block."""
    lines = md.split("\n")
    assert lines[0].startswith("# ")
    title = lines[0][2:].strip()
    i = md.index("## Abstract\n")
    j = md.index("\n## 1 Introduction")
    abstract = " ".join(md[i + len("## Abstract\n"):j].split())
    body = md[j + 1:]
    body = body.replace("figs/fig1_scale_curve.png", "figs/fig1_scale_curve.pdf")
    body = body.replace("figs/fig2_failure_by_size.png", "figs/fig2_failure_by_size.pdf")
    # bare web addresses (outside code spans) become autolinks, typeset with \url so they can break
    parts = re.split(r"(`[^`]*`)", body)
    url = re.compile(r"(?<![\w/<])((?:github\.com|arxiv\.org|honnibal\.dev|simonwillison\.net|rikiyat\.github\.io)/[^\s)\]]*[^\s)\].,;:])")
    body = "".join(x if x.startswith("`") else url.sub(r"<https://\1>", x) for x in parts)
    return title, abstract, body


def to_ascii(tex):
    for k, v in ASCII.items():
        tex = tex.replace(k, v)
    left = sorted({c for c in tex if ord(c) > 127})
    assert not left, f"non-ASCII left in main.tex: {left}"
    return tex


def main():
    title, abstract, body = split_front(open(os.path.join(PAPER, "preprint.md"), encoding="utf-8").read())
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(os.path.join(OUT, "figs"))
    for f in FIGS:
        shutil.copyfile(os.path.join(PAPER, "figs", f), os.path.join(OUT, "figs", f))
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "body.md")
        open(src, "w", encoding="utf-8").write(body)
        hdr = os.path.join(tmp, "header.tex")
        open(hdr, "w").write(HEADER)
        tex = subprocess.run(
            ["pandoc", src, "-f", FROM, "-t", "latex", "--standalone", "--listings", "--shift-heading-level-by=-1",
             "-H", hdr, "-M", f"title={title}", "-M", f"author={AUTHOR}", "-M", "date=November 2026",
             "-M", f"abstract={abstract}", "-V", "documentclass=article", "-V", "fontsize=10pt",
             "-V", "geometry:margin=1in", "-V", "colorlinks=true", "-V", "linkcolor=blue", "-V", "urlcolor=blue"],
            check=True, capture_output=True, text=True).stdout
    tex = to_ascii(tex)
    # pandoc numbers nothing (our headings carry their numbers); keep it that way under any class
    open(os.path.join(OUT, "main.tex"), "w").write(tex)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(OUT, "main.tex"), "main.tex")
        for f in FIGS:
            z.write(os.path.join(OUT, "figs", f), f"figs/{f}")
    with tempfile.TemporaryDirectory() as tmp:
        for name in ["main.tex"] + [f"figs/{f}" for f in FIGS]:
            os.makedirs(os.path.dirname(os.path.join(tmp, name)), exist_ok=True)
            shutil.copyfile(os.path.join(OUT, name), os.path.join(tmp, name))
        r = subprocess.run(["tectonic", "--keep-logs", "main.tex"], cwd=tmp, capture_output=True, text=True)
        log = open(os.path.join(tmp, "main.log")).read() if os.path.exists(os.path.join(tmp, "main.log")) else r.stderr
        if r.returncode:
            print(r.stderr[-3000:])
            raise SystemExit("tectonic failed")
        shutil.copyfile(os.path.join(tmp, "main.pdf"), PDF)
    over = re.findall(r"Overfull \\hbox \((\d+\.\d+)pt too wide\)[^\n]*\n", log)
    pages = re.search(r"Output written on .*?\((\d+) pages", log)
    print(f"main.tex {len(tex)} chars, ASCII; zip {os.path.getsize(ZIP) / 1e3:.0f} kB; pdf "
          f"{pages.group(1) if pages else '?'} pages; overfull boxes: {len(over)} (max {max(map(float, over), default=0):.0f}pt)")


if __name__ == "__main__":
    main()
