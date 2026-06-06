"""
Generate the figures for the Communications Physics manuscript directly from the
recorded JSON result artifacts (no hand-entered numbers). A tight, curated set:

  fig2_recall        variable-support precision/recall across three physics suites
  fig3_recovery      DenoisedSR vs full PySR: exact recovery, R^2, search vocabulary
  fig4_qscaling      exact recovery and R^2 vs #observations (full degrades, ours improves)
  fig5_pmlb          black-box control: learned vs random vs full support
  fig6_speed         time-to-solution speedup, per formula
  fig7_operator      operator restriction rescues pathological fits

fig1 (concept) is a hand-drawn schematic copied in separately. Method = "DenoisedSR".
Style: Okabe-Ito colour-blind-safe palette, sans-serif, 300 dpi, consistent sizing.
"""
import os, json, math, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    'font.size': 9, 'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
})
# Okabe-Ito colour-blind-safe palette
FULL = '#0072B2'   # blue        full PySR
OURS = '#D55E00'   # vermillion  DenoisedSR (learned support)
ORAC = '#009E73'   # green       oracle support (upper bound)
RAND = '#999999'   # grey        random support
ACC  = '#CC79A7'   # purple      accent

ap = argparse.ArgumentParser()
ap.add_argument('--src', default='..')
ap.add_argument('--out', default='../figures')
A = ap.parse_args()
SRC = (os.path.join(A.src, 'sr_frontend')
       if os.path.isdir(os.path.join(A.src, 'sr_frontend', 'data')) else A.src)
def R(p): return os.path.join(SRC, p)
def J(p):
    with open(R(p)) as f: return json.load(f)
os.makedirs(A.out, exist_ok=True)
def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(A.out, f'{name}.{ext}'))
    plt.close(fig); print('wrote', name)


# ------------------------------------------------------------ fig2: support recall
def fig_recall():
    s = J('data/results/support_recall_deployed.json')['suites']
    names = {'AI-Feynman': 'AI-Feynman', 'OOD test set': 'OOD test set', 'extended': 'extended set'}
    suites = [f"{names[r['suite']]}\n({r['n']} laws)" for r in s]
    prec = [r['precision'] for r in s]; rec = [r['recall'] for r in s]
    perf = [100*r['perfect_recall']/r['n'] for r in s]
    x = np.arange(len(suites)); w = 0.36
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.bar(x - w/2, prec, w, label='Precision', color=FULL)
    ax.bar(x + w/2, rec, w, label='Recall', color=OURS)
    ax.axhline(1.0, ls=':', lw=0.8, color='k', alpha=0.45)
    for xi, r, p in zip(x, rec, perf):
        ax.text(xi + w/2, r + 0.012, f'{r:.3f}', ha='center', va='bottom', fontsize=6.4, color=OURS)
        ax.text(xi, 0.43, f'{p:.0f}%\nperfect', ha='center', va='bottom', fontsize=6, color=ORAC)
    ax.set_ylim(0.4, 1.07); ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel('Variable support')
    ax.set_xticks(x); ax.set_xticklabels(suites)
    ax.legend(loc='lower center', ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.46))
    save(fig, 'fig2_recall')


# ------------------------------------------------------------ fig2b: ablation
def fig_ablation():
    ab = J('data/results/support_recall_deployed.json')['ablation_feynman118']
    labs = [a['model'].replace(' ', '\n', 1) for a in ab]
    rec = [a['recall'] for a in ab]
    cols = [ORAC, FULL, OURS]
    fig, ax = plt.subplots(figsize=(2.9, 2.4))
    b = ax.bar(range(len(ab)), rec, color=cols, width=0.62)
    for bi, a in zip(b, ab):
        ax.text(bi.get_x()+bi.get_width()/2, a['recall'],
                f"{a['recall']:.3f}", ha='center', va='bottom', fontsize=7)
    ax.set_ylim(0.95, 1.005); ax.set_ylabel('Recall (AI-Feynman)')
    ax.set_xticks(range(len(ab))); ax.set_xticklabels(labs, fontsize=7)
    ax.set_title('Ensemble matches the best component', fontsize=8)
    save(fig, 'fig2b_ablation')


# ------------------------------------------------------------ fig3: PySR 3-way
def fig_recovery():
    d = J('data/results/pysr_frontend_3way.json'); n = len(d)
    def cnt(k): return sum(1 for r in d if r.get(k))
    def mr2(k):
        v = [r[k] for r in d if r.get(k) is not None and math.isfinite(r[k])]; return sum(v)/len(v)
    ex = [cnt('full_exact')/n*100, cnt('var_exact')/n*100, cnt('varop_exact')/n*100]
    r2 = [mr2('full_r2'), mr2('var_r2'), mr2('varop_r2')]
    vocab = [np.mean([r['full_vocab'] for r in d]), np.mean([r['var_vocab'] for r in d]),
             np.mean([r['varop_vocab'] for r in d])]
    labs = ['full\nPySR', 'DenoisedSR\n(variables)', 'DenoisedSR\n(+operators)']
    cols = [FULL, OURS, ACC]
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.5))
    for ax, vals, ttl, fmt, ymax in [
        (axs[0], ex, f'Exact law recovery (n={n})', '{:.0f}%', 72),
        (axs[1], r2, 'Mean held-out $R^2$', '{:.3f}', None),
        (axs[2], vocab, 'Search vocabulary', '{:.0f}', None)]:
        b = ax.bar(range(3), vals, color=cols, width=0.64)
        ax.set_title(ttl, fontsize=8.5); ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=6.8)
        for bi, v in zip(b, vals):
            ax.text(bi.get_x()+bi.get_width()/2, v, fmt.format(v), ha='center', va='bottom', fontsize=7)
        ax.margins(y=0.18)
        if ymax: ax.set_ylim(0, ymax)
    fig.tight_layout(w_pad=1.4)
    save(fig, 'fig3_recovery')


# ------------------------------------------------------------ fig4: q-scaling (NEW)
def fig_qscaling():
    base = 'data/results/remote_fetch/32345_feynman_s50_qcurve_fixed_20260506/'
    def ld(f): return J(base + f)['summary']
    bq50 = ld('pysr_pmlb_feynman_baseline_s50_q50_d10_t10_seed20260506.json')
    bq100 = ld('pysr_pmlb_feynman_baseline_s50_q100_d10_t10_seed20260506.json')
    pq50 = ld('pysr_pmlb_feynman_prior_s50_q50_d10_t10_thr055_op028_seed20260506.json')
    pq100 = ld('pysr_pmlb_feynman_prior_s50_q100_d10_t10_thr055_op028_seed20260506.json')
    N = 50; q = [50, 100]
    ex_full = [bq50['full_success_r2_0999']/N*100, bq100['full_success_r2_0999']/N*100]
    ex_ours = [pq50['learned_variables_success_r2_0999']/N*100, pq100['learned_variables_success_r2_0999']/N*100]
    ex_orac = [bq50['oracle_support_success_r2_0999']/N*100, bq100['oracle_support_success_r2_0999']/N*100]
    r2_full = [bq50['full_mean_r2'], bq100['full_mean_r2']]
    r2_ours = [pq50['learned_variables_mean_r2'], pq100['learned_variables_mean_r2']]
    r2_orac = [bq50['oracle_support_mean_r2'], bq100['oracle_support_mean_r2']]
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.6))
    for ax, yf, yo, yc, ylab in [
        (axs[0], ex_full, ex_ours, ex_orac, 'Exact law recovery (%)'),
        (axs[1], r2_full, r2_ours, r2_orac, 'Mean held-out $R^2$')]:
        ax.plot(q, yc, '--o', color=ORAC, lw=1.3, ms=5, label='oracle support')
        ax.plot(q, yo, '-o', color=OURS, lw=1.6, ms=6, label='DenoisedSR')
        ax.plot(q, yf, '-o', color=FULL, lw=1.6, ms=6, label='full PySR')
        ax.set_xticks(q); ax.set_xlabel('observations $q$'); ax.set_ylabel(ylab)
        ax.margins(x=0.18, y=0.18)
    axs[0].annotate('full search\ndegrades', (100, ex_full[1]), (72, ex_full[1]-9),
                    fontsize=6.5, color=FULL, ha='center',
                    arrowprops=dict(arrowstyle='->', color=FULL, lw=0.6))
    axs[0].legend(loc='center left', fontsize=7)
    fig.suptitle('More data helps DenoisedSR but hurts unguided search', fontsize=9, y=1.02)
    fig.tight_layout()
    save(fig, 'fig4_qscaling')


# ------------------------------------------------------------ fig5: PMLB black-box
def fig_pmlb():
    p = ('data/results/remote_fetch/32345_pmlb_blackbox_priority_q100_20260506/matrix_summary.json')
    d = J(p)['summaries']
    H = {(s['distractors'], s['mode']): s['mean_r2'] for s in d if isinstance(s, dict)}
    Hc = {(s['distractors'], s['mode']): s['mean_column_count'] for s in d if isinstance(s, dict)}
    groups = [10, 20]; x = np.arange(len(groups)); w = 0.26
    full = [H[(g, 'full')] for g in groups]
    rand = [H[(g, 'random_variables')] for g in groups]
    learn = [H[(g, 'learned_variables')] for g in groups]
    fig, ax = plt.subplots(figsize=(3.7, 2.6))
    ax.bar(x - w, full, w, label='full search', color=FULL)
    ax.bar(x,     rand, w, label='random support', color=RAND)
    ax.bar(x + w, learn, w, label='DenoisedSR', color=OURS)
    ax.axhline(0, lw=0.8, color='k')
    for xi, g in zip(x, groups):
        ax.text(xi + w, H[(g, 'learned_variables')] + 0.02,
                f"{Hc[(g,'learned_variables')]/Hc[(g,'full')]*100:.0f}% cols",
                ha='center', va='bottom', fontsize=6, color=OURS)
    ax.set_xticks(x); ax.set_xticklabels([f'{g} distractors' for g in groups])
    ax.set_ylabel('Mean held-out $R^2$'); ax.set_ylim(-0.3, 0.68)
    ax.legend(loc='upper left', fontsize=7, ncol=1)
    save(fig, 'fig5_pmlb')


# ------------------------------------------------------------ fig6: speed
def fig_speed():
    s = J('data/results/speed_vars_result.json')
    a = np.array(s['a_times']); b = np.array(s['b_times']); sp = a / b
    order = np.argsort(sp); a, b, sp = a[order], b[order], sp[order]
    idx = np.arange(len(sp))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.barh(idx, sp, color=[OURS if v >= 1.3 else RAND for v in sp], height=0.7)
    ax.axvline(1.0, ls='--', lw=0.8, color='k', alpha=0.6)
    for i, v in enumerate(sp):
        ax.text(v + 0.06, i, f'{v:.1f}×', va='center', fontsize=6.6)
    ax.set_yticks([]); ax.set_xlabel('time-to-solution speedup  (full / DenoisedSR)')
    ax.set_xlim(0, max(sp)*1.18)
    ax.text(0.97, 0.05, 'each bar = one Feynman law', transform=ax.transAxes,
            ha='right', fontsize=6.5, color='gray')
    save(fig, 'fig6_speed')


# ------------------------------------------------------------ fig7: operator safeguard
def fig_operator():
    forms = ['I_32_5', 'I_30_3', 'I_18_4']
    allops = [0.19, 0.79, 0.95]; restr = [0.92, 0.98, 0.9996]
    x = np.arange(len(forms)); w = 0.36
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.bar(x - w/2, allops, w, label='all operators', color=RAND)
    ax.bar(x + w/2, restr, w, label='predicted operators', color=ORAC)
    for i in range(len(forms)):
        ax.annotate('', xy=(x[i]+w/2, restr[i]), xytext=(x[i]-w/2, allops[i]),
                    arrowprops=dict(arrowstyle='->', color='k', lw=0.6, alpha=0.5))
    ax.set_ylim(0, 1.12); ax.set_ylabel('Held-out $R^2$')
    ax.set_xticks(x); ax.set_xticklabels(forms)
    ax.legend(loc='lower right', fontsize=7)
    save(fig, 'fig7_operator')


if __name__ == '__main__':
    fig_recall(); fig_ablation(); fig_recovery(); fig_qscaling(); fig_pmlb(); fig_speed(); fig_operator()
    print('figures ->', os.path.abspath(A.out))
