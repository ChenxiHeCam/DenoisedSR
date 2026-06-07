"""Fetch SRSD-Feynman dummy variants (easy/medium/hard) train files from HF.
Stores raw .txt under data/benchmarks/srsd/{split}_dummy/<formula>.txt
plus a manifest mapping formula -> (true vars, dummy vars, sympy_eq_str).
"""
import os, json, urllib.request, urllib.parse, re
from pathlib import Path

ROOT = Path("data/benchmarks/srsd")
ROOT.mkdir(parents=True, exist_ok=True)
BASE = "https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_{split}_dummy/resolve/main/train/{name}.txt"

def list_keys(split):
    with open(ROOT/f"{split}_supp_info.json") as f: d = json.load(f)
    return list(d.keys()), d

manifest = {}
for split in ["easy","medium","hard"]:
    keys, supp = list_keys(split)
    out_dir = ROOT/f"{split}_dummy"; out_dir.mkdir(exist_ok=True)
    print(f"=== {split}: {len(keys)} formulas ===")
    for k in keys:
        target = out_dir/f"{k}.txt"
        if target.exists() and target.stat().st_size > 1024:
            continue
        url = BASE.format(split=split, name=k)
        try:
            urllib.request.urlretrieve(url, target)
        except Exception as e:
            print(f"  FAIL {k}: {e}")
            continue
    n_ok = sum(1 for k in keys if (out_dir/f"{k}.txt").exists())
    print(f"  downloaded {n_ok}/{len(keys)}")
    for k in keys:
        info = supp[k]
        # parse vars from sympy_eq_str
        eq = info["sympy_eq_str"]
        used = set(re.findall(r"x\d+", eq))
        dummies = set(info.get("dummy_vars", []))
        manifest[k] = {
            "split": split,
            "file": str((out_dir/f"{k}.txt").as_posix()),
            "sympy_eq_str": eq,
            "true_vars": sorted(used),
            "dummy_vars": sorted(dummies),
            "n_true": len(used),
            "n_dummy": len(dummies),
        }
(Path(ROOT)/"manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"\nManifest: {len(manifest)} entries -> {ROOT/'manifest.json'}")
