"""
Generate all figures for the Communications Physics manuscript directly from the
recorded JSON result artifacts (no hand-entered numbers). Run from repo root of
papers/sr_frontend so the relative result paths resolve, or pass --src.

Figures produced (PDF + PNG, 300 dpi) into ../figures/:
  fig2_support_recall      variable-support precision/recall across suites
  fig3_pysr_recovery       Full vs +variables vs +operators: exact, R^2, columns
  fig4_speed               per-formula time-to-solution speedup
  fig5_operator_safeguard  pathological-fit rescue by operator restriction
"""
import os, json, math, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    'font.size': 9, 'font.family': 'sans-serif', 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})
# Nature-portfolio-ish palette
C = {'full': '#4575b4', 'var': '#d73027', 'varop': '#fc8d59',
     'random': '#999999', 'oracle': '#1a9850', 'accent': '#542788'}

ap = argparse.ArgumentParser()
ap.add_argument('--src', default='..')  # path to sr_frontend (results live there)
ap.add_argument('--out', default='../figures')
A = ap.parse_args()
SRC = os.path.join(A.src, 'sr_frontend') if os.path.isdir(os.path.join(A.src, 'sr_frontend', 'data')) else A.src
def R(p): return os.path.join(SRC, p)
os.makedirs(A.out, exist_ok=True)
def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(A.out, f'{name}.{ext}'))
    plt.close(fig); print('wrote', name)


def jload(p):
    with open(R(p)) as f: return json.load(f)


# ---------------------------------------------------------------- Fig 2: support
def fig_support():
    # large-suite numbers (eval_support_predictor / eval_v5_suite / eval_feynman_suite)
    suites = ['AI-Feynman\n(118)', 'real591\n(91)', 'v5 physics\n(1085)']
    prec = [1.000, 0.959, 0.999]
    rec  = [1.000, 1.000, 0.999]
    perfect = [100.0, 100.0, 99.7]
    x = np.arange(len(suites)); w = 0.38
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.bar(x - w/2, prec, w, label='Precision', color=C['full'])
    ax.bar(x + w/2, rec, w, label='Recall', color=C['var'])
    for i, p in enumerate(perfect):
        ax.text(x[i], 1.02, f'{p:.0f}%\nperfect', ha='center', va='bottom', fontsize=6.5, color=C['oracle'])
    ax.set_ylim(0, 1.18); ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.set_ylabel('Variable support'); ax.set_xticks(x); ax.set_xticklabels(suites)
    ax.axhline(1.0, ls=':', lw=0.7, color='k', alpha=0.4)
    ax.legend(loc='lower center', ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.42))
    save(fig, 'fig2_support_recall')


# ---------------------------------------------------------------- Fig 3: PySR 3-way
def fig_pysr():
    d = jload('data/results/pysr_frontend_3way.json'); n = len(d)
    def cnt(k): return sum(1 for r in d if r.get(k))
    def mr2(k):
        v = [r[k] for r in d if r.get(k) is not None and math.isfinite(r[k])]
        return sum(v)/len(v)
    ex = [cnt('full_exact')/n*100, cnt('var_exact')/n*100, cnt('varop_exact')/n*100]
    r2 = [mr2('full_r2'), mr2('var_r2'), mr2('varop_r2')]
    vocab = [np.mean([r['full_vocab'] for r in d]), np.mean([r['var_vocab'] for r in d]),
             np.mean([r['varop_vocab'] for r in d])]
    labs = ['Full\nPySR', '+variable\nprior', '+operator\nprior']
    cols = [C['full'], C['var'], C['varop']]
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.4))
    for ax, vals, ttl, fmt in [
        (axs[0], ex, f'Exact recovery (n={n})', '{:.0f}%'),
        (axs[1], r2, 'Mean $R^2$', '{:.3f}'),
        (axs[2], vocab, 'Search vocabulary', '{:.0f}')]:
        b = ax.bar(range(3), vals, color=cols, width=0.66)
        ax.set_title(ttl, fontsize=8.5); ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=7)
        for bi, v in zip(b, vals):
            ax.text(bi.get_x()+bi.get_width()/2, v, fmt.format(v), ha='center', va='bottom', fontsize=7)
        ax.margins(y=0.18)
    axs[0].set_ylim(0, 75)
    fig.tight_layout()
    save(fig, 'fig3_pysr_recovery')


# ---------------------------------------------------------------- Fig 4: speed
def fig_speed():
    s = jload('data/results/speed_vars_result.json')
    a = np.array(s['a_times']); b = np.array(s['b_times'])
    sp = a / b
    order = np.argsort(-sp)
    a, b, sp = a[order], b[order], sp[order]
    idx = np.arange(len(sp))
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.barh(idx, sp, color=[C['var'] if x >= 1.3 else C['random'] for x in sp], height=0.7)
    ax.axvline(1.0, ls='--', lw=0.8, color='k', alpha=0.6)
    for i, v in enumerate(sp):
        ax.text(v + 0.05, i, f'{v:.1f}×', va='center', fontsize=6.5)
    ax.set_yticks([]); ax.set_xlabel('Time-to-solution speedup (full / pruned)')
    ax.set_xlim(0, max(sp)*1.18)
    ax.text(0.98, 0.04, 'each bar = one Feynman formula', transform=ax.transAxes,
            ha='right', fontsize=6.5, color='gray')
    save(fig, 'fig4_speed')


# ---------------------------------------------------------------- Fig 5: operator safeguard
def fig_op():
    forms = ['I_32_5', 'I_30_3', 'I_18_4']
    allops = [0.19, 0.79, 0.95]; restr = [0.92, 0.98, 0.9996]
    x = np.arange(len(forms)); w = 0.38
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    ax.bar(x - w/2, allops, w, label='all operators', color=C['random'])
    ax.bar(x + w/2, restr, w, label='restricted (predicted)', color=C['oracle'])
    for i in range(len(forms)):
        ax.annotate('', xy=(x[i]+w/2, restr[i]), xytext=(x[i]-w/2, allops[i]),
                    arrowprops=dict(arrowstyle='->', color='k', lw=0.6, alpha=0.5))
    ax.set_ylim(0, 1.1); ax.set_ylabel('Held-out $R^2$'); ax.set_xticks(x); ax.set_xticklabels(forms)
    ax.set_title('Operator restriction rescues pathological fits', fontsize=8)
    ax.legend(loc='lower right', fontsize=7)
    save(fig, 'fig5_operator_safeguard')


if __name__ == '__main__':
    fig_support(); fig_pysr(); fig_speed(); fig_op()
    print('all figures ->', os.path.abspath(A.out))
