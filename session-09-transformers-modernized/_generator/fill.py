import os, re, subprocess, sys, pathlib
HERE=os.path.dirname(os.path.abspath(__file__))
D="session-09-transformers-modernized"; tex=pathlib.Path(D)/"slides-transformers-modernized.tex"
s=tex.read_text()
want=set(sys.argv[1:])  # e.g. "l3:3-8"
s=re.sub(r"% FRAMES-BEGIN (l3|l4) (\d+) (\d+)\n.*?% FRAMES-END\n", lambda m: f"% FRAMES {m.group(1)} {m.group(2)} {m.group(3)}\n" if f"{m.group(1)}:{m.group(2)}-{m.group(3)}" in want else m.group(0), s, flags=re.S)
for m in list(re.finditer(r"% FRAMES (l3|l4) (\d+) (\d+)\n", s))[::-1]:
    key=f"{m.group(1)}:{m.group(2)}-{m.group(3)}"
    if key not in want: continue
    src=m.group(1); pdf=f"{D}/figure/cs336-{src}.pdf"
    out=subprocess.run(["uv","run","--no-project","-q","--with","pymupdf","python",
        HERE+"/gen_frames.py",pdf,f"figure/cs336-{src}.pdf",m.group(2),m.group(3)],
        capture_output=True,text=True,check=True).stdout
    s=s[:m.start()]+f"% FRAMES-BEGIN {src} {m.group(2)} {m.group(3)}\n"+out+"% FRAMES-END\n"+s[m.end():]
tex.write_text(s)
