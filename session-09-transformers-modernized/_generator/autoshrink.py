"""Build, find overflowing CS336 frames, record [shrink] overrides, refill. Repeat until clean."""
import os, json, re, subprocess, sys, pathlib
HERE=os.path.dirname(os.path.abspath(__file__))
S=HERE; D="session-09-transformers-modernized"
keys=sys.argv[1:]
ovp=pathlib.Path(S)/"overrides.json"; ov=json.loads(ovp.read_text()) if ovp.exists() else {}
for it in range(4):
    subprocess.run(["python3",f"{S}/fill.py",*keys],check=True,timeout=300)
    subprocess.run(["bash",f"{S}/build.sh"],check=True)
    log=open(f"{D}/slides-transformers-modernized.log").read(); tex=open(f"{D}/slides-transformers-modernized.tex").read().splitlines()
    hits={}
    for m in re.finditer(r"Overfull \\vbox \(([0-9.]+)pt too high\) detected at line (\d+)",log):
        amt=float(m.group(1)); ln=int(m.group(2))
        if amt<2: continue
        sec=None
        for l in reversed(tex[:ln]):
            if l.startswith("% FRAMES-BEGIN"): sec=l.split()[2]; break
        sl=next((l for l in reversed(tex[:ln]) if l.startswith("% CS336 slide")),None)
        if sl and sec: hits[f"{sec}:{sl.split()[-1]}"]=amt
    print("iteration",it,"overflowing:",hits)
    if not hits: break
    for k,amt in hits.items():
        cur=ov.get(k,{}).get("shrink",0)
        ov.setdefault(k,{})["shrink"]=min(40,cur+int(amt/200*100)+6)
    ovp.write_text(json.dumps(ov,indent=1))
print("overrides:",ov)
