"""Turn CS336 slides into beamer frames: title + source text as bullets + vector crops
of figures/formulas (\\includegraphics[page=N,trim=...,clip]). Usage:
    gen_frames.py <pdf> <figure-relpath> <first> <last> > frames.tex
"""
import sys, json, os
import pymupdf as fitz

OVR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "overrides.json")
OVR = json.load(open(OVR_PATH)) if os.path.exists(OVR_PATH) else {}

PAGE_W, PAGE_H = 720, 405
SCALE = 0.6           # source pt -> frame pt for crops (frame is ~0.63 of the slide width)
MIN_SCALE = 0.33      # below this, show the whole slide body as one crop instead
MATH_FONTS = ("CambriaMath",)


def esc(t):
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}"),
                 ("–", "--"), ("—", "---"), ("‘", "`"), ("’", "'"), ("“", "``"), ("”", "''"), ("…", r"\ldots{}"),
                 ("→", r"$\rightarrow$"), ("≈", r"$\approx$"), ("×", r"$\times$"), ("≥", r"$\geq$"), ("≤", r"$\leq$")]:
        t = t.replace(a, b)
    return t


def merge(rects, pad=8):
    rects = [list(r) for r in rects]
    changed = True
    while changed:
        changed = False
        out = []
        while rects:
            r = rects.pop()
            for o in out:
                if not (r[2] + pad < o[0] or o[2] + pad < r[0] or r[3] + pad < o[1] or o[3] + pad < r[1]):
                    o[:] = [min(r[0], o[0]), min(r[1], o[1]), max(r[2], o[2]), max(r[3], o[3])]
                    changed = True
                    break
            else:
                out.append(r)
        rects = out
    return rects


def inside(b, r, tol=3):
    return b[0] >= r[0] - tol and b[1] >= r[1] - tol and b[2] <= r[2] + tol and b[3] <= r[3] + tol


def overlaps(b, r):
    return not (b[2] <= r[0] or r[2] <= b[0] or b[3] <= r[1] or r[3] <= b[1])


def slide_frame(pg, n, figpath, src="l3"):
    ovr = OVR.get(f"{src}:{n}", {})
    d = pg.get_text("dict")
    blocks = []
    for b in d["blocks"]:
        if b["type"]:
            continue
        spans = [s for l in b["lines"] for s in l["spans"]]
        text = " ".join("".join(s["text"] for s in l["spans"]) for l in b["lines"]).strip()
        if not text:
            continue
        blocks.append({"bbox": list(b["bbox"]), "text": " ".join(text.split()),
                       "size": max(s["size"] for s in spans),
                       "math": any(s["font"].startswith(MATH_FONTS) for s in spans)})
    title = next((b for b in blocks if b["size"] >= 22 and b["bbox"][1] < 70), None)
    body = [b for b in blocks if b is not title and b["bbox"][1] >= 64]

    vis = [list(r["bbox"]) for r in pg.get_image_info()]
    for dr in pg.get_drawings():
        r = dr["rect"]
        if r.y1 <= 30 or (r.width > 0.9 * PAGE_W and r.height > 0.9 * PAGE_H):
            continue
        vis.append([r.x0, r.y0, r.x1, r.y1])
    vis += [b["bbox"] for b in body if b["math"]]
    vis = [v for v in vis if v[3] > 64]
    regions = merge(vis)
    # a region that swallows most of the body: crop the whole body instead
    regions = [[max(r[0], 0), max(r[1], 64), min(r[2], PAGE_W), min(r[3], PAGE_H)] for r in regions]
    # credits: small text right below/inside a region
    text_blocks = [b for b in body if not any(inside(b["bbox"], r) for r in regions)]
    partial = [b for b in text_blocks if any(overlaps(b["bbox"], r) for r in regions)]
    for b in partial:  # text crossing a region joins it (keeps layouts intact)
        for r in regions:
            if overlaps(b["bbox"], r):
                r[:] = [min(r[0], b["bbox"][0]), min(r[1], b["bbox"][1]), max(r[2], b["bbox"][2]), max(r[3], b["bbox"][3])]
    regions = merge(regions, pad=2)
    def near(b, r, d=40):
        x0, y0, x1, y1 = b
        return not (x1 + d < r[0] or r[2] + d < x0 or y1 + d < r[1] or r[3] + d < y0)
    for blk in body:
        if len(blk["text"]) <= 30 and any(near(blk["bbox"], r) for r in regions) \
                and not any(inside(blk["bbox"], r) for r in regions):
            r = next(r for r in regions if near(blk["bbox"], r))
            bb = blk["bbox"]
            r[:] = [min(r[0], bb[0]), min(r[1], bb[1]), max(r[2], bb[2]), max(r[3], bb[3])]
    regions = merge(regions, pad=2)
    text_blocks = [b for b in body if not any(inside(b["bbox"], r) for r in regions)]
    text_blocks.sort(key=lambda b: (round(b["bbox"][1] / 6), b["bbox"][0]))

    ttl = esc(title["text"]) if title else f"Slide {n}"
    if ovr.get("title"):
        ttl = ovr["title"]
    opt = f"[shrink={ovr['shrink']}]" if ovr.get("shrink") else ""
    out = [f"% CS336 slide {n}", f"\\begin{{frame}}{opt}{{{ttl}}}"]
    items = []
    BUL = "•❖▪◦"
    for b in text_blocks:
        raw = b["text"]
        for old, new in ovr.get("replace", []):   # remove course-internal references (listed in order.txt)
            raw = raw.replace(old, new)
        for ch in BUL[1:]:
            raw = raw.replace(ch, "•")
        starts_with_bullet = raw.lstrip().startswith("•")
        pieces = [x.strip(" -") for x in raw.split("•")]
        pieces = [x for x in pieces if x]
        lvl = 2 if b["bbox"][0] > 100 else 1
        small = b["size"] < 12
        for k, t in enumerate(pieces):
            plvl = lvl if (k == 0 or starts_with_bullet) else min(lvl + 1, 2)
            items.append((plvl, small, esc(t)))
    left_text = text_blocks and max(b["bbox"][2] for b in text_blocks) < 400
    right_figs = regions and min(r[0] for r in regions) > 340
    side = bool(items and left_text and right_figs)

    def item_lines():
        lines, open2 = ["\\begin{itemize}"], False
        seen1 = False
        for lvl, small, t in items:
            if small:
                continue
            if lvl == 2 and not seen1:   # a sub-item needs a parent item: promote
                lvl = 1
            seen1 = seen1 or lvl == 1
            if lvl == 2 and not open2:
                lines.append("  \\begin{itemize}"); open2 = True
            if lvl == 1 and open2:
                lines.append("  \\end{itemize}"); open2 = False
            lines.append(("    " if lvl == 2 else "  ") + "\\item " + t)
        if open2:
            lines.append("  \\end{itemize}")
        lines.append("\\end{itemize}")
        return lines if len(lines) > 2 else []

    AVAIL_H = 165.0   # usable frame height below the title, in pt

    def fig_lines(maxw, text_h):
        lines = []
        regs = sorted(regions, key=lambda r: (r[1], r[0]))
        total_h = sum(r[3] - r[1] + 6 for r in regs) or 1
        scale = min(SCALE, max(AVAIL_H - text_h, 60) / total_h)
        for r in regs:
            x0, y0, x1, y1 = r[0] - 3, r[1] - 3, r[2] + 3, r[3] + 3
            trim = f"{x0:.0f} {PAGE_H - y1:.0f} {PAGE_W - x1:.0f} {y0:.0f}"
            w = min((x1 - x0) * scale, maxw)
            lines.append(f"  \\includegraphics[page={n},trim={trim},clip,width={w:.0f}pt]{{{figpath}}}\\par")
        return lines

    credits = [t for lvl, small, t in items if small]

    def eff_scale(maxw, text_h):
        if not regions:
            return 1.0
        total_h = sum(r[3] - r[1] + 6 for r in regions) or 1
        sc = min(SCALE, max(AVAIL_H - text_h, 60) / total_h)
        return min(min(sc, maxw / (r[2] - r[0] + 6)) for r in regions)
    nb = sum(1 for _, sm, _ in items if not sm)
    sc = eff_scale(200, 0) if side else eff_scale(330, 16.0 * nb + (12 if nb else 0))
    if regions and (sc < MIN_SCALE or ovr.get("fullbody")):
        x0, y0, x1, y1 = 20, 64, 712, 402
        trim = f"{x0} {PAGE_H - y1} {PAGE_W - x1} {y0}"
        w = min(330, (AVAIL_H + 3) / (y1 - y0) * (x1 - x0))
        out += ["% full-body crop (figures would be too small next to the bullets)", "\\begin{center}",
                f"  \\includegraphics[page={n},trim={trim},clip,width={w:.0f}pt]{{{figpath}}}", "\\end{center}", "\\end{frame}\n"]
        return "\n".join(out)
    if side:
        out += ["\\begin{columns}[T]", "\\begin{column}{0.5\\textwidth}"] + item_lines() + \
               ["\\end{column}", "\\begin{column}{0.48\\textwidth}", "\\centering"] + fig_lines(200, 0) + \
               ["\\end{column}", "\\end{columns}"]
    else:
        out += item_lines()
        if regions:
            nb = sum(1 for _, sm, _ in items if not sm)
            out += ["\\begin{center}"] + fig_lines(330, 16.0 * nb + (12 if nb else 0)) + ["\\end{center}"]
    if credits:
        out.append("{\\scriptsize " + "; ".join(credits) + "\\par}")
    out.append("\\end{frame}\n")
    return "\n".join(out)


if __name__ == "__main__":
    pdf, figpath, a, b = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    doc = fitz.open(pdf)
    for n in range(a, b + 1):
        src = "l4" if "l4" in os.path.basename(pdf) else "l3"
        print(slide_frame(doc[n - 1], n, figpath, src))
