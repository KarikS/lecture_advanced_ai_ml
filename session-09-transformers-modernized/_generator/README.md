# Generator for slides-transformers-modernized.tex

Run from the repository root (needs `uv`; PyMuPDF is pulled in a throwaway environment, not a repo dependency):

    python3 session-09-transformers-modernized/_generator/skeleton.py      # rewrites the .tex skeleton incl. [O] and License frames
    python3 session-09-transformers-modernized/_generator/autoshrink.py l3:3-8 l3:9-14 l3:15-19 l3:20-29 l3:30-35 \
        l3:36-52 l3:53-57 l3:58-58 l3:59-68 l4:2-12 l4:13-23 l4:24-37 l4:38-47

`overrides.json` holds per-slide adjustments (`title`, `shrink`, `fullbody`); `autoshrink.py`
adds `shrink` entries for frames that overflow. Hand edits to the .tex are lost on regeneration —
put them in `skeleton.py` (non-source frames) or `overrides.json` (source frames).
