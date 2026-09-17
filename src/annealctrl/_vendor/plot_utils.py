# Vendored copy of research-os/scripts/plot_utils.py.
#
# Upstream: ~/research-os/scripts/plot_utils.py
# Upstream revision: untracked
# sha256 of the vendored source below: e0357c77fb65d32ad7826aaa8e2e8b3d22273c034bb8cd202ef9b6efa9f08992
# Vendored on: 2026-09-17
#
# Why a copy lives here: figures must not embed Type 3 fonts (ADR-0006), and a
# public repository whose figures cannot be regenerated on a clean machine is a
# reproducibility gap. annealctrl.figures prefers the upstream module whenever it
# is present and only falls back to this copy, so the canonical file stays
# canonical. Do not edit this copy; re-vendor from upstream instead.

"""Camera-ready matplotlib defaults for ML conference submissions.

Why this exists
---------------
The installed `matplotlib` skill is explicit that it renders "Tim's personal
aesthetic": whitegrid, DejaVu Sans, dpi=150 (style-reference.md:36-38). That is a
blog aesthetic, not a camera-ready one. More seriously, no skill on this machine
-- not `matplotlib`, not `figure-designer`, not `ml-paper-writing`, and not
`pre-submission-reviewer`, which claims to audit "figure quality" -- ever sets
`pdf.fonttype`. Matplotlib's default embeds **Type 3** fonts, which IEEE PDF
eXpress rejects outright and NeurIPS/ICML/CVPR flag. You can pass every review
gate in the v1 workflow and still be stopped by the submission system.

`fonttype = 42` means TrueType. It is one line, and it is the whole fix.

Usage
-----
    import sys; sys.path.insert(0, str(Path.home() / "research-os/scripts"))
    from plot_utils import use_venue, save

    use_venue("neurips", column="single")
    fig, ax = plt.subplots()
    ...
    save(fig, "figures/ablation")     # writes .pdf and .png, then verifies fonts
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# Text width in inches. From each venue's own style file, so a figure saved at
# this width drops into the column at scale=1.0 and its fonts stay at the size
# you set instead of being silently shrunk by \includegraphics.
_VENUE = {
    "neurips": {"single": 5.50, "double": 5.50, "base": 10},
    "icml":    {"single": 3.25, "double": 6.75, "base": 9},
    "iclr":    {"single": 5.50, "double": 5.50, "base": 10},
    "acl":     {"single": 3.15, "double": 6.30, "base": 9},
    "cvpr":    {"single": 3.25, "double": 6.875, "base": 8},
    "aaai":    {"single": 3.30, "double": 7.00, "base": 9},
}

# Okabe-Ito: colourblind-safe and still legible in greyscale print.
OKABE_ITO = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
]


def use_venue(venue: str = "neurips", column: str = "single", base: int | None = None) -> float:
    """Install venue-correct rcParams. Returns the text width in inches."""
    venue = venue.lower()
    if venue not in _VENUE:
        raise ValueError(f"unknown venue {venue!r}; known: {', '.join(sorted(_VENUE))}")
    spec = _VENUE[venue]
    width = spec[column]
    size = base or spec["base"]

    matplotlib.rcParams.update({
        # The whole point of this module.
        "pdf.fonttype": 42,     # TrueType, not Type 3
        "ps.fonttype": 42,
        "pdf.compression": 6,

        # Match the body font of the paper rather than the plotting library's.
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",

        "font.size": size,
        "axes.labelsize": size,
        "axes.titlesize": size,
        "xtick.labelsize": size - 1,
        "ytick.labelsize": size - 1,
        "legend.fontsize": size - 1,

        "figure.figsize": (width, width * 0.62),
        "figure.dpi": 200,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,

        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "errorbar.capsize": 2,
        "axes.prop_cycle": matplotlib.cycler(color=OKABE_ITO),
    })
    return width


def _scan_type3(pdf: Path) -> bool | None:
    """Pure-Python Type 3 scan. True = found, False = clean, None = inconclusive.

    Font descriptors live in object dictionaries, which are usually outside the
    compressed content streams, so a raw scan finds /Type3 reliably. Object
    streams (PDF 1.5+) can hide them, so also try inflating every FlateDecode
    stream before concluding the file is clean.
    """
    try:
        raw = pdf.read_bytes()
    except OSError:
        return None
    if b"/Type3" in raw or b"/Subtype /Type3" in raw:
        return True
    import re
    import zlib

    saw_stream = False
    for m in re.finditer(rb"stream\r?\n", raw):
        saw_stream = True
        chunk = raw[m.end() : m.end() + 400_000]
        end = chunk.find(b"endstream")
        try:
            if b"/Type3" in zlib.decompress(chunk[: end if end > 0 else None]):
                return True
        except zlib.error:
            continue
    return False if saw_stream or raw else None


def check_fonts(pdf: Path) -> tuple[bool, str]:
    """Return (ok, report). ok is False if any Type 3 font is embedded."""
    pdf = Path(pdf)
    if not pdf.exists():
        return False, f"{pdf} does not exist"

    try:
        out = subprocess.run(["pdffonts", str(pdf)], capture_output=True,
                             text=True, timeout=30).stdout
        bad = [ln for ln in out.splitlines() if "Type 3" in ln]
        if bad:
            return False, "Type 3 fonts embedded (IEEE PDF eXpress rejects these):\n" + "\n".join(bad)
        return True, "no Type 3 fonts (pdffonts)"
    except (FileNotFoundError, subprocess.SubprocessError):
        pass  # poppler absent; fall through to the built-in scan

    found = _scan_type3(pdf)
    if found is True:
        return False, "Type 3 fonts embedded (IEEE PDF eXpress rejects these) [built-in scan]"
    if found is False:
        return True, "no Type 3 fonts [built-in scan; `brew install poppler` for the authoritative check]"
    return False, "could not read the PDF to check fonts"


def save(fig, stem: str | Path, formats: tuple[str, ...] = ("pdf", "png")) -> list[Path]:
    """Save, then verify. A figure that silently embeds Type 3 is the bug."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for f in formats:
        p = stem.with_suffix(f".{f}")
        fig.savefig(p)
        written.append(p)
    pdfs = [p for p in written if p.suffix == ".pdf"]
    if pdfs:
        ok, report = check_fonts(pdfs[0])
        print(f"[plot_utils] {pdfs[0]}: {report}")
        if not ok:
            raise RuntimeError(f"refusing to ship {pdfs[0]}: {report}")
    return written


if __name__ == "__main__":
    import numpy as np

    w = use_venue("neurips", "single")
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 60)
    fig, ax = plt.subplots()
    for i, lbl in enumerate(["HEC-GNN", "GCN", "baseline"]):
        ax.plot(x, np.sin(x + i) + rng.normal(0, 0.05, x.size), label=lbl)
    ax.set_xlabel("chain strength")
    ax.set_ylabel("energy gap")
    ax.legend()
    out = save(fig, "/tmp/plot_utils_selftest")
    print(f"[plot_utils] venue width = {w}in; wrote {', '.join(str(p) for p in out)}")
