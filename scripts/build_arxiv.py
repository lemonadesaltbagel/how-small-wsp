"""Build the arXiv source from paper/preprint.md: paper/arxiv/{main.tex, figs/*.pdf}, compiled by pdflatex (arXiv's
engine). Title, author and abstract go into the LaTeX front matter; figures use the PDF versions; non-ASCII characters
are mapped in a small header so pdflatex accepts them.
  python3 -I scripts/build_arxiv.py        (needs pandoc; compile with: cd paper/arxiv && pdflatex main && pdflatex main)"""
import os
import re
import shutil
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "paper", "preprint.md")
OUT = os.path.join(ROOT, "paper", "arxiv")

HEADER = r"""
\usepackage{newunicodechar}
\usepackage{etoolbox}
\newunicodechar{−}{\ensuremath{-}}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{±}{\ensuremath{\pm}}
\newunicodechar{·}{\ensuremath{\cdot}}
\newunicodechar{δ}{\ensuremath{\delta}}
\newunicodechar{β}{\ensuremath{\beta}}
\newunicodechar{ᵀ}{\ensuremath{^{\top}}}
\newunicodechar{¥}{\textyen{}}
\usepackage{textcomp}
\usepackage{microtype}
\usepackage{float}
\let\origfigure\figure
\let\endorigfigure\endfigure
\renewenvironment{figure}[1][]{\origfigure[H]}{\endorigfigure}
\renewcommand{\arraystretch}{1.15}
\AtBeginEnvironment{longtable}{\footnotesize}
\AtBeginEnvironment{Shaded}{\footnotesize}
\AtBeginEnvironment{verbatim}{\footnotesize}
"""


def main():
    md = open(SRC, encoding="utf-8").read()
    lines = md.split("\n")
    title = lines[0].lstrip("# ").strip()
    author = lines[2].split("·")[0].strip()
    date = lines[2].split("·")[1].split(".")[0].strip()
    a0 = md.index("## Abstract\n")
    a1 = md.index("\n## 1 ")
    abstract = md[a0 + len("## Abstract\n"):a1].strip()
    body = md[a1 + 1:]
    body = body.replace("d̂", "$\\hat{d}$")
    body = re.sub(r"\]\(figs/([^)]+)\.png\)", r"](figs/\1.pdf)", body)
    # the figure captions are the paragraphs after each image; pandoc's implicit figure caption is the alt text
    body = re.sub(r"!\[(Figure \d+)\]\((figs/[^)]+)\)\n\n(Figure \d+: .*?)\n\n",
                  lambda m: f"![{re.sub(r'^Figure \d+: ', '', m.group(3)).replace(chr(10), ' ')}]({m.group(2)})\n\n",
                  body, flags=re.S)
    meta = (f"---\ntitle: \"{title}\"\nauthor: \"{author}\"\ndate: \"{date}\"\nabstract: |\n"
            + "\n".join("  " + ln for ln in abstract.split("\n")) + "\n---\n\n")
    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
    for f in os.listdir(os.path.join(ROOT, "paper", "figs")):
        if f.endswith(".pdf"):
            shutil.copyfile(os.path.join(ROOT, "paper", "figs", f), os.path.join(OUT, "figs", f))
    with tempfile.TemporaryDirectory() as tmp:
        src, hdr = os.path.join(tmp, "main.md"), os.path.join(tmp, "header.tex")
        open(src, "w", encoding="utf-8").write(meta + body)
        open(hdr, "w", encoding="utf-8").write(HEADER)
        subprocess.run(["pandoc", src, "-s", "-f", "markdown", "-t", "latex",
                        "--shift-heading-level-by=-1", "--include-in-header", hdr, "--columns=100",
                        "-V", "documentclass=article", "-V", "fontsize=10pt", "-V", "geometry:margin=1in",
                        "-V", "colorlinks=true", "-V", "linkcolor=blue", "-V", "urlcolor=blue",
                        "-o", os.path.join(OUT, "main.tex")], check=True)
    print("wrote", os.path.relpath(os.path.join(OUT, "main.tex"), ROOT))


if __name__ == "__main__":
    main()
