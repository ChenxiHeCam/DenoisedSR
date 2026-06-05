"""Show all non-exact cases (R2<0.9999) with high R2 for human inspection."""
import json
from pathlib import Path

rows = json.loads(Path("data/results/numeric_equiv_rows.json").read_text())

# Categorize
high_r2_nonexact = [r for r in rows if 0.90 <= r["r2"] < 0.9999]
mid_r2 = [r for r in rows if 0.5 <= r["r2"] < 0.90]
fail = [r for r in rows if r["r2"] < 0.5]
exact = [r for r in rows if r["r2"] >= 0.9999]

print(f"Total: {len(rows)} formulas")
print(f"  Exact (R2>=0.9999):       {len(exact)}")
print(f"  High R2 non-exact (0.9-1): {len(high_r2_nonexact)}  <- spurious candidates")
print(f"  Mid R2 (0.5-0.9):          {len(mid_r2)}")
print(f"  Fail (R2<0.5):             {len(fail)}")
print()
print("="*100)
print("HIGH R2 BUT NOT EXACT — human-judge whether equivalent:")
print("="*100)
for r in sorted(high_r2_nonexact, key=lambda x:-x["r2"]):
    print(f"\n[{r['law_id']}]  R2={r['r2']:.5f}")
    print(f"  TRUTH: {r['truth']}")
    print(f"  vars(x0..xN): {r['sel']}")
    print(f"  FOUND: {r['expr']}")
