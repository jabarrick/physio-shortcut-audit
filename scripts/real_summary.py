"""Flat real-audit summary + trial-type tables (PILOT_LOG 17.25). Read-only on results/real."""
import glob, json, csv
from collections import defaultdict, Counter
def ba(c):
    tp=fn=fp=tn=0
    for v in c.values(): a,b,cc,dd=v; tp+=a; fn+=b; fp+=cc; tn+=dd
    r1 = tp/(tp+fn) if tp+fn else float('nan'); r0 = tn/(tn+fp) if tn+fp else float('nan'); return (r1+r0)/2
rows=[]
for f in sorted(glob.glob("results/real/*__seed?.json")):
    d=json.load(open(f)); b=ba(d["counts"])
    rows.append({"model":d["model"],"fold":d["fold"],"seed":d["seed"],"ba":d["ba"],"ba_counts":b,
      "sri_post":ba(d["sri"]["post_ctrl"])-ba(d["sri"]["post_alpha"]),"sri_central":ba(d["sri"]["central_ctrl"])-ba(d["sri"]["central_alpha"]),
      "heog_reg":b-ba(d["heog"]),"ig_alpha_share":d["ig"]["alpha"]["share"],"ig_alpha_norm_abs":d["ig"]["alpha"]["norm_abs"],
      "ig_saccade_share":d["ig"]["saccade"]["share"],"ig_saccade_norm_abs":d["ig"]["saccade"]["norm_abs"],
      "alpha_erasure_keep":d["alpha_erasure"]["delta_ba_keep"],"alpha_probe_D":d["alpha_probe_D"],
      "best_val_ba":d["training"]["best_val_ba"],"epochs":d["training"]["epochs"]})
with open("results/analysis/real_summary.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# trial types (main arm, unique trials) and accuracy by type x arm x model
types=Counter(); per_sub=defaultdict(Counter); acc=defaultdict(lambda:[0,0])
seen=set()
for f in glob.glob("results/real/*_trials.csv"):
    for r in csv.DictReader(open(f)):
        key=(r["subject"],r["run"],r["trial"])
        if r["arm"]=="main" and key not in seen:
            seen.add(key); types[r["type"]]+=1; per_sub[r["subject"]][r["type"]]+=1
        a=acc[(r["model"],r["arm"],r["type"])]; a[0]+=int(r["correct"]); a[1]+=1
out={"trial_types":dict(types),"n_trials":sum(types.values()),"n_subjects":len(per_sub),
     "per_subject":{s:dict(v) for s,v in per_sub.items()},
     "accuracy":{"|".join(k):{"correct":v[0],"n":v[1],"acc":v[0]/v[1]} for k,v in acc.items()}}
json.dump(out,open("results/analysis/real_trials_summary.json","w"),indent=1)
print(len(rows), out["trial_types"], out["n_trials"], out["n_subjects"])
