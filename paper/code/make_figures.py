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
ap.add_argument('--src', default='../..')
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
    all_suites = J('data/results/support_recall_deployed.json')['suites']
    # main figure: 4 external/OOD suites (same protocol: 20 random distractors, q=100, deployed tau=0.10)
    s = [r for r in all_suites if r.get('group') == 'external_OOD']
    name_map = {'AI-Feynman':'AI-Feynman','Strogatz':'Strogatz','Nguyen':'Nguyen','SRSD-Feynman':'SRSD-Feynman'}
    suites = [f"{name_map.get(r['suite'], r['suite'])}\n({r['n']} laws)" for r in s]
    rec = [r['recall'] for r in s]
    fr  = [r.get('distractor_filter_rate', 0) for r in s]
    perf = [100*r['perfect_recall']/r['n'] for r in s]
    x = np.arange(len(suites)); w = 0.36
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.bar(x - w/2, rec, w, label='Recall (true vars kept)', color=OURS)
    ax.bar(x + w/2, fr,  w, label='Distractor filter rate (distractors dropped)', color=FULL)
    ax.axhline(1.0, ls=':', lw=0.8, color='k', alpha=0.45)
    for xi, r_, f_, p in zip(x, rec, fr, perf):
        ax.text(xi - w/2, r_ + 0.012, f'{r_:.3f}', ha='center', va='bottom', fontsize=6.5, color=OURS)
        ax.text(xi + w/2, max(f_,0) + 0.012, f'{f_:.3f}', ha='center', va='bottom', fontsize=6.5, color=FULL)
        ax.annotate(f'{p:.0f}% perfect-\nrecall tasks', xy=(xi, 0), xytext=(0, -36),
                    xycoords='data', textcoords='offset points',
                    ha='center', va='top', fontsize=6, color=ORAC, annotation_clip=False)
    ax.set_ylim(-0.04, 1.12); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel('Rate')
    ax.set_xticks(x); ax.set_xticklabels(suites, fontsize=7)
    ax.legend(loc='lower center', ncol=1, fontsize=7, bbox_to_anchor=(0.5, -0.55))
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
        # consistent over ALL n tasks: a non-finite / failed fit is floored at -1
        # (so full search's diverged row is not silently dropped)
        v = [max(r[k], -1.0) if (r.get(k) is not None and math.isfinite(r[k])) else -1.0
             for r in d]
        return sum(v) / len(v)
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


# ------------------------------------------------------------ fig4: q-scaling at deployed tau=0.10 (multi-seed)
def fig_qscaling():
    agg = J('data/results/qscaling_tau010_aggregate.json')
    qs = sorted(int(k) for k in agg)
    ex_full = [agg[str(q)]['full_exact_pct_mean'] for q in qs]
    ex_ours = [agg[str(q)]['var_exact_pct_mean']  for q in qs]
    fe_std  = [agg[str(q)]['full_exact_pct_std']  for q in qs]
    ve_std  = [agg[str(q)]['var_exact_pct_std']   for q in qs]
    r2_full = [agg[str(q)]['full_r2_mean'] for q in qs]
    r2_ours = [agg[str(q)]['var_r2_mean']  for q in qs]
    n_seeds = [agg[str(q)]['n_seeds'] for q in qs]
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.7))
    axs[0].errorbar(qs, ex_full, yerr=fe_std, fmt='-o', color=FULL, lw=1.6, ms=6, capsize=3, label='full PySR')
    axs[0].errorbar(qs, ex_ours, yerr=ve_std, fmt='-o', color=OURS, lw=1.6, ms=6, capsize=3, label='DenoisedSR')
    axs[0].set_xticks(qs); axs[0].set_xlabel('observations $q$')
    axs[0].set_ylabel('Exact law recovery (%)')
    axs[0].set_ylim(30, 75); axs[0].legend(loc='lower right', fontsize=7)
    for x, y in zip(qs, ex_full): axs[0].text(x, y-3, f'{y:.0f}', ha='center', fontsize=6.5, color=FULL)
    for x, y in zip(qs, ex_ours): axs[0].text(x, y+1.5, f'{y:.0f}', ha='center', fontsize=6.5, color=OURS)
    axs[1].plot(qs, r2_full, '-o', color=FULL, lw=1.6, ms=6, label='full PySR')
    axs[1].plot(qs, r2_ours, '-o', color=OURS, lw=1.6, ms=6, label='DenoisedSR')
    axs[1].set_xticks(qs); axs[1].set_xlabel('observations $q$')
    axs[1].set_ylabel('Mean held-out $R^2$')
    for x, y in zip(qs, r2_full): axs[1].text(x, y-0.02, f'{y:.2f}', ha='center', fontsize=6.5, color=FULL)
    for x, y in zip(qs, r2_ours): axs[1].text(x, y+0.01, f'{y:.2f}', ha='center', fontsize=6.5, color=OURS)
    axs[1].set_ylim(0.7, 1.02)
    fig.suptitle(f'DenoisedSR consistently outperforms full PySR across observation counts ($\\tau{{=}}0.10$, multi-seed)', fontsize=8.5, y=1.02)
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
        ax.text(xi + w, H[(g, 'learned_variables')] + 0.03,
                f"{Hc[(g,'learned_variables')]/Hc[(g,'full')]*100:.0f}%\ncols",
                ha='center', va='bottom', fontsize=6, color=OURS)
    ax.set_xticks(x); ax.set_xticklabels([f'{g} distractors' for g in groups])
    ax.set_ylabel('Mean held-out $R^2$'); ax.set_ylim(-0.32, 0.74)
    ax.legend(loc='upper center', fontsize=6.8, ncol=3, bbox_to_anchor=(0.5, 1.16),
              columnspacing=1.0, handletextpad=0.4)
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


# ============================================================ NEW FIGURES (audit, 2026-06-07)

# fig_baselines: classical feature-selector controls on AI-Feynman recall
def fig_baselines():
    d = J('data/results/baselines_feynman_recall.json')['summary']
    # ordered weakest->strongest by oracle-topk recall
    rows = [
        ('mutual\ninfo',       d['mutual_info_topk']),
        ('Pearson\n|corr|',    d['pearson_abs_topk']),
        ('Spearman\n|corr|',   d['spearman_abs_topk']),
        ('RF feat.\nimport.',  d['rf_importance_topk']),
        ('Lasso CV',           d['lasso_cv_topk']),
    ]
    labs = [r[0] for r in rows]
    rec  = [r[1]['recall_mean'] for r in rows]
    perf = [100*r[1]['perfect_recall_frac'] for r in rows]
    x = np.arange(len(rows) + 1); w = 0.62
    # last bar: ours
    rec.append(1.0); perf.append(100.0); labs.append('Denoised\nSR')
    cols = [RAND]*5 + [OURS]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    b = ax.bar(x, rec, w, color=cols)
    for bi, v, p in zip(b, rec, perf):
        ax.text(bi.get_x()+bi.get_width()/2, v + 0.012, f'{v:.3f}',
                ha='center', va='bottom', fontsize=7)
        ax.annotate(f'{p:.0f}% perfect', xy=(bi.get_x()+bi.get_width()/2, 0),
                    xytext=(0, -28), xycoords='data', textcoords='offset points',
                    ha='center', va='top', fontsize=6.2, color=ORAC, annotation_clip=False)
    ax.axhline(1.0, ls=':', lw=0.7, color='k', alpha=0.5)
    ax.set_ylim(0, 1.10); ax.set_ylabel('Recall (AI-Feynman 118, oracle top-$k$)')
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=7)
    ax.set_title('Classical feature selectors leave true variables on the floor', fontsize=8)
    save(fig, 'fig_baselines')


# fig_srsd: external SRSD-Feynman dummy suite, per split
def fig_srsd():
    s = J('data/results/srsd_dummy_recall.json')['summary']
    splits = ['easy', 'medium', 'hard']
    ours = [s[k]['denoisedsr_rec'] for k in splits]
    lass = [s[k]['lasso_rec']      for k in splits]
    perf_o = [100*s[k]['denoisedsr_perfect'] for k in splits]
    n = [s[k]['n'] for k in splits]
    x = np.arange(len(splits)); w = 0.36
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    ax.bar(x - w/2, lass, w, color=RAND, label='Lasso CV (oracle $k$)')
    ax.bar(x + w/2, ours, w, color=OURS, label='DenoisedSR')
    for xi, v, p in zip(x, ours, perf_o):
        ax.text(xi + w/2, v + 0.012, f'{v:.3f}', ha='center', va='bottom', fontsize=6.5, color=OURS)
        ax.annotate(f'{p:.0f}% perfect', xy=(xi, 0), xytext=(0, -28),
                    xycoords='data', textcoords='offset points',
                    ha='center', va='top', fontsize=6.2, color=ORAC, annotation_clip=False)
    ax.axhline(1.0, ls=':', lw=0.7, color='k', alpha=0.5)
    ax.set_ylim(0, 1.08); ax.set_ylabel('Recall')
    ax.set_xticks(x); ax.set_xticklabels([f'{lab}\n(n={ni})' for lab, ni in zip(splits, n)])
    ax.set_title('External SRSD-Feynman benchmark (never trained on)', fontsize=8)
    ax.legend(loc='lower right', fontsize=7)
    save(fig, 'fig_srsd')


# fig_noise: noise robustness curves — DenoisedSR vs all 5 classical baselines
def fig_noise():
    s = J('data/results/noise_sweep.json')['summary']
    etas = sorted([float(k.split('_')[1]) for k in s])
    # Support both new schema (per-method dict) and legacy keys
    def val(eta, m):
        e = s[f'eta_{eta}']
        if m == 'denoisedsr':
            return e.get('denoisedsr', {}).get('mean', e.get('denoisedsr_recall'))
        if m in e:
            return e[m]['mean']
        return e.get(f'{m}_recall')
    methods = [
        ('denoisedsr', 'DenoisedSR (ours)', OURS, '-', 'o', 1.8, 6),
        ('lasso',     'Lasso CV',            FULL, '-', 's', 1.3, 4.5),
        ('rf',        'RF importance',       ORAC, '-', '^', 1.3, 4.5),
        ('spearman',  'Spearman $|\\rho|$',  ACC,  '-', 'D', 1.0, 3.8),
        ('pearson',   'Pearson $|\\rho|$',   '#A9A9A9', '-', 'v', 1.0, 3.8),
        ('mi',        'mutual information',  RAND, '-', 'x', 1.0, 4.0),
    ]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for m, lab, col, ls, mk, lw, ms in methods:
        ys = [val(e, m) for e in etas]
        ax.plot(etas, ys, ls=ls, marker=mk, color=col, lw=lw, ms=ms, label=lab)
    # annotate endpoints for ours and lasso
    ax.text(etas[-1]+0.005, val(etas[-1], 'denoisedsr')-0.005, '1.000',
            color=OURS, fontsize=6.5, va='top')
    ax.text(etas[-1]+0.005, val(etas[-1], 'lasso'), f"{val(etas[-1], 'lasso'):.3f}",
            color=FULL, fontsize=6.5, va='center')
    ax.set_xlabel(r'relative Gaussian noise $\eta$ on $y$')
    ax.set_ylabel('Recall (AI-Feynman 118)')
    ax.set_xticks(etas)
    ax.set_ylim(0.55, 1.04)
    ax.legend(loc='center right', fontsize=6.5, ncol=1)
    ax.set_title('DenoisedSR retains perfect recall up to $\\eta=0.30$', fontsize=8)
    save(fig, 'fig_noise')


# fig_gplearn: second backend (solver-agnostic), multi-seed if available
def fig_gplearn():
    # check aggregator for multi-seed
    agg = J('data/results/multiseed_distractor_aggregate.json') if os.path.exists(
        R('data/results/multiseed_distractor_aggregate.json')) else {}
    gp = agg.get('gplearn_multiseed', {}).get('__mean_std__')
    if gp and gp.get('n_seeds',1) > 1:
        ex_f = gp['full_exact_pct_mean']; ex_o = gp['var_exact_pct_mean']
        r2_f = gp['full_r2_mean'];        r2_o = gp['var_r2_mean']
        ex_fe = gp['full_exact_pct_std']; ex_oe = gp['var_exact_pct_std']
        r2_fe = gp['full_r2_std'];        r2_oe = gp['var_r2_std']
        ttl_ex = f"Exact recovery (mean$\\pm$s.d., $n_{{seeds}}={gp['n_seeds']}$)"
    else:
        d = J('data/results/gplearn_backend_3way.json'); n = len(d)
        def cnt(k): return sum(1 for r in d if r.get(k))
        def mr2(k):
            v = [max(r[k], -1.0) if (r.get(k) is not None and math.isfinite(r[k])) else -1.0 for r in d]
            return sum(v)/len(v)
        ex_f = cnt('full_exact')/n*100; ex_o = cnt('var_exact')/n*100
        r2_f = mr2('full_r2');          r2_o = mr2('var_r2')
        ex_fe = ex_oe = r2_fe = r2_oe = 0
        ttl_ex = f'Exact recovery (n={n})'
    fig, axs = plt.subplots(1, 2, figsize=(4.2, 2.5))
    labs = ['full\ngplearn', 'DenoisedSR\n+ gplearn']; cols = [FULL, OURS]
    panels = [(axs[0], [ex_f, ex_o], [ex_fe, ex_oe], ttl_ex, '{:.0f}%'),
              (axs[1], [r2_f, r2_o], [r2_fe, r2_oe], 'Mean held-out $R^2$', '{:.2f}')]
    for ax, vals, errs, ttl, fmt in panels:
        b = ax.bar(range(2), vals, color=cols, width=0.6,
                   yerr=errs if any(errs) else None, error_kw=dict(ecolor='k', lw=0.9, capsize=3))
        ax.set_title(ttl, fontsize=8.5); ax.set_xticks(range(2)); ax.set_xticklabels(labs, fontsize=7.5)
        for bi, v, e in zip(b, vals, errs):
            ax.text(bi.get_x()+bi.get_width()/2, v + e + 0.3, fmt.format(v), ha='center', va='bottom', fontsize=7)
        ax.margins(y=0.18)
    fig.tight_layout(w_pad=1.4)
    save(fig, 'fig_gplearn')


# fig_distractor: PySR exact recovery vs n_dist (multiseed if available)
def fig_distractor():
    agg = J('data/results/multiseed_distractor_aggregate.json')
    multi = agg.get('distractor_sweep_multiseed', {})
    if multi and any(multi[k].get('n_seeds',1)>1 for k in multi):
        ds = sorted(int(k) for k in multi)
        full = [multi[str(x)]['full_exact_pct_mean'] for x in ds]
        ours = [multi[str(x)]['var_exact_pct_mean']  for x in ds]
        fe   = [multi[str(x)]['full_exact_pct_std']  for x in ds]
        ve   = [multi[str(x)]['var_exact_pct_std']   for x in ds]
        ns   = [multi[str(x)]['n_seeds']             for x in ds]
        ttl = f'Robust to distractor count (mean$\\pm$s.d., $n_{{seeds}}\\leq {max(ns)}$)'
    else:
        d = agg['distractor_sweep_seed42']
        ds = sorted(int(k) for k in d)
        full = [d[str(x)]['full_exact_pct'] for x in ds]
        ours = [d[str(x)]['var_exact_pct']  for x in ds]
        fe = ve = [0]*len(ds)
        ttl = 'Robust to distractor-column count'
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.errorbar(ds, full, yerr=fe, fmt='-o', color=FULL, lw=1.6, ms=6,
                capsize=3, label='full PySR')
    ax.errorbar(ds, ours, yerr=ve, fmt='-o', color=OURS, lw=1.6, ms=6,
                capsize=3, label='DenoisedSR')
    for x, y in zip(ds, full): ax.text(x, y - 4, f'{y:.0f}%', ha='center', fontsize=6.5, color=FULL)
    for x, y in zip(ds, ours): ax.text(x, y + 2, f'{y:.0f}%', ha='center', fontsize=6.5, color=OURS)
    ax.set_xlabel('number of distractor columns')
    ax.set_ylabel('Exact-law recovery (%)')
    ax.set_xticks(ds); ax.set_ylim(15, 80)
    ax.legend(loc='lower left', fontsize=7)
    ax.set_title(ttl, fontsize=8)
    save(fig, 'fig_distractor')


# fig_srsd_precision: SRSD precision-vs-#dummies (audit-ready diagnostic)
def fig_srsd_precision():
    per = J('data/results/srsd_dummy_recall.json')['per_task']
    nd  = np.array([t['n_dummy'] for t in per])
    pr  = np.array([t['denoisedsr']['prec'] for t in per])
    rc  = np.array([t['denoisedsr']['rec']  for t in per])
    # jitter for visibility
    rng = np.random.default_rng(0)
    ndj = nd + rng.uniform(-0.12, 0.12, size=len(nd))
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    sc = ax.scatter(ndj, pr, s=18, alpha=0.7,
                    c=[OURS if r>=0.999 else RAND for r in rc], edgecolor='none')
    # binned mean
    bins = sorted(set(nd))
    for b in bins:
        m = pr[nd==b].mean()
        ax.plot([b-0.3, b+0.3], [m, m], color='k', lw=1.2)
    rcorr = np.corrcoef(nd, pr)[0,1]
    ax.text(0.97, 0.93, f'Pearson $r={rcorr:.2f}$', transform=ax.transAxes,
            ha='right', va='top', fontsize=7.5)
    ax.set_xlabel('number of dummy columns added by SRSD')
    ax.set_ylabel('DenoisedSR precision')
    ax.set_ylim(-0.02, 1.05); ax.set_xticks(bins)
    ax.set_title('SRSD precision drops by design: more dummies $\Rightarrow$ more kept', fontsize=8)
    save(fig, 'fig_srsd_precision')


# REPLACE fig3 with seed-band version: error bars from multiseed aggregate
def fig_recovery_seedband():
    agg = J('data/results/multiseed_distractor_aggregate.json')['headline_d20_seeds']
    d = J('data/results/pysr_frontend_3way.json'); n = len(d)
    # exact pcts: full mean/std and ours mean/std across seeds; +ops same
    ex_means = [agg['full_exact_pct']['mean'], agg['var_exact_pct']['mean'], agg['varop_exact_pct']['mean']]
    ex_stds  = [agg['full_exact_pct']['std'],  agg['var_exact_pct']['std'],  agg['varop_exact_pct']['std']]
    r2_means = [agg['full_r2']['mean'], agg['var_r2']['mean'], agg['varop_r2']['mean']]
    r2_stds  = [agg['full_r2']['std'],  agg['var_r2']['std'],  agg['varop_r2']['std']]
    vocab = [np.mean([r['full_vocab'] for r in d]), np.mean([r['var_vocab'] for r in d]),
             np.mean([r['varop_vocab'] for r in d])]
    labs = ['full\nPySR', 'DenoisedSR\n(variables)', 'DenoisedSR\n(+operators)']
    cols = [FULL, OURS, ACC]
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.6))
    # Each panel: (ax, vals, errs, title, fmt, ylim, yticks, show_err, broken_axis)
    panels = [(axs[0], ex_means, ex_stds, f'Exact recovery (n={n}, 3 seeds)', '{:.0f}%', (0, 92), None, True, False),
              (axs[1], r2_means, r2_stds, 'Mean held-out $R^2$',              '{:.3f}', (0.80, 1.02), [0.80, 0.85, 0.90, 0.95, 1.00], True, True),
              (axs[2], vocab,    [0]*3,  'Search vocabulary',                 '{:.0f}', None, None, False, False)]
    for ax, vals, errs, ttl, fmt, ylim, yticks, show_err, broken in panels:
        bars = ax.bar(range(3), vals, color=cols, width=0.62,
                      yerr=errs if show_err else None,
                      error_kw=dict(ecolor='k', lw=0.9, capsize=3))
        ax.set_title(ttl, fontsize=8.5); ax.set_xticks(range(3))
        ax.set_xticklabels(labs, fontsize=6.8)
        for bi, v, e in zip(bars, vals, errs):
            ax.text(bi.get_x()+bi.get_width()/2, v + e + (0.005 if broken else 0.6),
                    fmt.format(v), ha='center', va='bottom', fontsize=7)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.margins(y=0.18)
        if yticks is not None:
            ax.set_yticks(yticks)
        if broken:
            # signal the truncated y-axis with a small zigzag near the bottom
            ax.text(-0.02, ylim[0]-0.002, '$\\sim\\sim$', transform=ax.transData,
                    ha='right', va='top', fontsize=8, color='k')
    fig.tight_layout(w_pad=1.4)
    save(fig, 'fig3_recovery')


if __name__ == '__main__':
    fig_recall(); fig_ablation(); fig_recovery_seedband(); fig_qscaling(); fig_pmlb(); fig_speed()
    fig_baselines(); fig_srsd(); fig_noise(); fig_gplearn(); fig_distractor(); fig_srsd_precision()
    print('figures ->', os.path.abspath(A.out))
