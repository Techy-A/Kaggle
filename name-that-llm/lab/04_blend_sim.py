"""Step 4: simulate the 0.556 BLEND (not one member) on a test-like pool, and calibrate it against known LB results (CPU).

Steps 1-3 scored changes on the stylegram alone; both changes they backed failed on the LB (stylegram-add 0.550, the
GLM+Claude gate 0.555), because the other members already carry much of the same signal. This rebuilds most of the
0.556 notebook's blend from honest out-of-fold (OOF) predictions and applies it exactly as the notebook does:

  members (notebook weight W):   stylegram 1.516 (our 5 prompt-grouped folds; plain SVC + full-group centered twin)
                                 lora 0.177, lora2 0.113 (public fold runs: per-fold validation features x readouts)
                                 tokattn 0.153, tokattn_deb 0.185 (public OOF probabilities)
                                 debv3l 0.05, electl 0.032 (heads refit in our folds on public train encodings)
  centered twins on clustered rows: q3_8b 1.5 x 0.219, glm9b 1.5 x 0.110, debv3l 0.25, lora 0.15
  not available, left out:        ultrabert 0.095, desk 0.132, lora3 0.177, style_lgb 0.063, the q25_7b/style4b twins
  test-time steps: the LM heads' test-set standardization, the notebook's mixglm clustering (TF-IDF + GLM-4-9B L24,
                   cut 0.60) for the twins, the stylegram's own clustering (cut 0.90), Sinkhorn balancing (alpha),
                   and the within-Claude gate that picks WHICH Claude after the blend (stylegram SVC on the fold's Claude
                   rows + the public Claude API passages, lambda 1.5; its LoRA Claude head is not in the fold readouts).

Every OOF prediction comes from a model that never saw that passage's prompt (all fold schemes hold out whole prompts),
so a pseudo-test pool of a few passages per prompt drawn from all 150 prompts behaves like the hidden test. Calibration
(18 Oct): 2-3 per prompt gets the centering signs wrong; 4-6 per prompt reproduces all five LB signs below.

Variants and their known public LB scores (the calibration targets):
  base        0.556 notebook: stylegram twin REPLACES the plain head on its clustered rows      LB 0.556
  no_center   no stylegram centering at all (the 0.552 notebook)                              LB 0.552
  add         stylegram twin ADDED to the plain head (beta 1)                                 LB 0.550
  alpha025    Sinkhorn alpha 0.25 instead of 0.5                                              LB 0.556
  glm_gate    + GLM+Claude gate on glm9b L8+L3 (lambda 1)                                     LB 0.555
The sim is trusted only if it orders these the way the LB did; then new ideas are tested against 'base'.

Usage: python 04_blend_sim.py [--draws 10] [--sizes 4 5 6] [--screen] [--v559] [--ref base]
"""
import argparse
import glob
import importlib.util
import os
import re

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import vstack
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

HERE = os.path.dirname(os.path.abspath(__file__))
OOF = os.path.join(HERE, 'data/oof')
API = os.path.join(HERE, 'data/ntl_extra_passages_claude.csv')   # seyitkaangunes/ntl-extra-api-passages (no train prompts)
CLAUDE_GATE_LAM = 1.5
spec = importlib.util.spec_from_file_location('sim', os.path.join(HERE, '01_stylegram_sim.py'))
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)   # the notebook's stylegram, sparse centering, stylegram clustering, SVC probabilities

W = {'stylegram': 1.516, 'lora': 0.177, 'lora2': 0.113, 'tokattn': 0.153, 'tokattn_deb': 0.185,
     'debv3l': 0.05, 'electl': 0.032}
TWIN_W = {'q3_8b': 1.5 * 0.219, 'glm9b': 1.5 * 0.110, 'debv3l': 0.25, 'lora': 0.15}


def softmax_log(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def group_center(X, groups):
    X = np.asarray(X, dtype=np.float32).copy()
    for g in np.unique(groups):
        i = groups == g
        X[i] -= X[i].mean(0)
    return X


def mixed_clusters(texts, E, thresh=0.60):
    """The notebook's mixglm clustering: mean of content-word TF-IDF and GLM-4-9B L24 cosine distances, average linkage."""
    X = TfidfVectorizer(stop_words='english', sublinear_tf=True).fit_transform(texts)
    Dt = np.clip(1 - (X @ X.T).toarray(), 0, None)
    Z = np.asarray(E, dtype=np.float64)
    Z = Z - Z.mean(0)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    D = (Dt + np.clip(1 - Z @ Z.T, 0, None)) / 2
    np.fill_diagonal(D, 0)
    return fcluster(linkage(squareform(D, checks=False), method='average'), thresh, criterion='distance')


def lr(C):
    return LogisticRegression(C=C, max_iter=5000)


def lora_oof(pattern, n, idx_of, K):
    """OOF plain log-probabilities and the raw features of a fold-trained LoRA (each row from the model that held it out),
    plus that fold's centered readout for the twin."""
    feats = np.zeros((n, 32768), np.float32)
    lp = np.zeros((n, K))
    ro_of = [None] * n
    for d in sorted(glob.glob(pattern)):
        v = np.load(glob.glob(d + '/valid*features.npz')[0], allow_pickle=True)
        ro = np.load(d + '/readouts.npz')
        rows = np.array([idx_of[i] for i in v['ids']])
        f = v['features'].astype(np.float32)
        feats[rows] = f
        lp[rows] = softmax_log(f @ ro['plain_coef'].T + ro['plain_intercept'])
        for r in rows:
            ro_of[r] = ro
    assert all(r is not None for r in ro_of), pattern
    return lp, feats, ro_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--draws', type=int, default=10)
    ap.add_argument('--screen', action='store_true')   # also score the 18 Oct screening ideas (plain LM heads, scales, half)
    ap.add_argument('--v559', action='store_true')   # also score the public 0.559 notebook's changes, one at a time
    ap.add_argument('--ref', default='base')   # variant the paired deltas are measured against
    ap.add_argument('--sizes', type=int, nargs='+', default=[4, 5, 6])   # passages per prompt; 4-6 reproduces every LB sign, 2-3 does not
    args = ap.parse_args()

    train = pd.read_csv(os.path.join(HERE, 'data/train.csv'))
    labels = sorted(train.model.unique())
    K, n = len(labels), len(train)
    y = train.model.map({l: i for i, l in enumerate(labels)}).values
    pid = train.prompt_id.to_numpy(dtype=object)
    texts = train.text.to_numpy(dtype=object)
    idx_of = {i: r for r, i in enumerate(train.id.values)}

    # ---- public OOF members ----
    plain = {}
    for key, f in [('tokattn', 'tokattn/tokattn_oof.npz'), ('tokattn_deb', 'tokattn2/tokattn_debl23_oof.npz')]:
        z = np.load(os.path.join(OOF, f), allow_pickle=True)
        assert (z['ids'] == train.id.values).all() and list(z['classes']) == labels
        plain[key] = np.log(np.clip(z['probs'], 1e-9, 1))
    plain['lora'], lora_feats, lora_ro = lora_oof(os.path.join(OOF, 'lora/lora24/outer[0-4]'), n, idx_of, K)
    plain['lora2'], _, _ = lora_oof(os.path.join(OOF, 'lora2/lora_out/job_[0-4]'), n, idx_of, K)
    print('loaded public OOF members', flush=True)

    # ---- features for heads refit in our folds ----
    dz = np.load(os.path.join(OOF, 'enc/enc_debv3l2_train.npz'), allow_pickle=True)
    ez = np.load(os.path.join(OOF, 'enc/enc_electl_train.npz'), allow_pickle=True)
    li = lambda z, L: list(z['layers']).index(L)
    E = {'debv3l': np.hstack([dz['emb'][:, li(dz, 24)], dz['emb'][:, li(dz, 6)]]).astype(np.float32),
         'electl': np.hstack([ez['emb'][:, li(ez, 21)], ez['emb_style'][:, li(ez, 24)]]).astype(np.float32)}
    fz = np.load(os.path.join(HERE, 'data/featcache/ntl_feats.npz'), allow_pickle=True)
    E['q3_8b'] = fz['q3_8b'].astype(np.float32)
    E['glm9b'] = fz['glm9b'][:, :8192].astype(np.float32)
    glm_l24 = fz['glm9b'][:, 8192:].astype(np.float32)
    X = sim.stylegram_features(texts.tolist())
    api = pd.read_csv(API)
    # ponytail: one vocabulary over train + API text for the Claude gate (the notebook fits it on Claude + API rows only)
    XA = sim.stylegram_features(texts.tolist() + api.text.astype(str).tolist())
    XA_tr, XA_api = XA[:n], XA[n:]
    y_api = api.model.map({l: i for i, l in enumerate(labels)}).values
    cl_idx = np.array([i for i, l in enumerate(labels) if l.startswith('claude')])
    if args.v559:   # the 0.559 notebook's family gates: train rows + external rows at weight 0.25, balanced LinearSVC
        prv = pd.read_csv(os.path.join(HERE, 'data/prvsiyan/passages_clean.csv'))   # CC BY 4.0, no train prompts
        prv = prv[prv.model.isin(labels) & prv.text.notna()]
        ext_y = np.concatenate([y_api, prv.model.map({l: i for i, l in enumerate(labels)}).values])
        XB = sim.stylegram_features(texts.tolist() + api.text.astype(str).tolist() + prv.text.astype(str).tolist())
        XB_tr, XB_ext = XB[:n], XB[n:]
        lab_idx = lambda pre: np.array([i for i, l in enumerate(labels) if l.startswith(pre)])
        FAM = [('claude', lab_idx('claude'), 0.5), ('gpt', lab_idx('gpt'), 1.0), ('gemini', lab_idx('gemini'), 1.0),
               ('glm', lab_idx('glm'), 0.5)]

    fold = np.empty(n, int)
    for k, (_, va) in enumerate(GroupKFold(5).split(X, y, pid)):
        fold[va] = k
    gc_idx = np.array([i for i, l in enumerate(labels) if l.startswith(('glm', 'claude'))])
    M = []
    for k in range(5):
        tr = fold != k
        m = dict(svc=LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(X[tr], y[tr]),
                 svc_c=LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(sim.center(X[tr], pid[tr]), y[tr]))
        for key, C in [('debv3l', 0.005), ('electl', 0.005)] + ([('q3_8b', 0.0025), ('glm9b', 0.0025)] if args.screen else []):
            sc = StandardScaler().fit(E[key][tr])
            m[key] = lr(C).fit(sc.transform(E[key][tr]), y[tr])
        for key, C in [('q3_8b', 0.0025), ('glm9b', 0.0025), ('debv3l', 0.005)]:   # centered twins
            Xc = group_center(E[key][tr], pid[tr])
            csc = StandardScaler().fit(Xc)
            m[f'{key}_c'] = (csc, lr(C).fit(csc.transform(Xc), y[tr]))
        g = tr & np.isin(y, gc_idx)   # GLM+Claude gate, train scaling (as submitted)
        gsc = StandardScaler().fit(E['glm9b'][g])
        m['gate'] = (gsc, lr(0.0025).fit(gsc.transform(E['glm9b'][g]), y[g]))
        c = tr & np.isin(y, cl_idx)
        m['claude'] = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(vstack([XA_tr[c], XA_api]),
                                                                             np.concatenate([y[c], y_api]))
        for fam, fidx, C in (FAM if args.v559 else []):   # API rows are all Claude, so only the Claude gate gets them
            r, e = tr & np.isin(y, fidx), np.isin(ext_y, fidx)
            m[f'fam_{fam}'] = LinearSVC(C=C, class_weight='balanced', random_state=42, max_iter=4000).fit(
                vstack([XB_tr[r], XB_ext[e]]), np.concatenate([y[r], ext_y[e]]),
                sample_weight=np.r_[np.ones(r.sum()), np.full(e.sum(), 0.25)])
        M.append(m)
        print(f'fold {k} fitted', flush=True)

    res = {}
    for d in range(args.draws):
        rng = np.random.default_rng(100 + d)
        rows = np.concatenate([rng.choice(np.flatnonzero(pid == p), int(rng.choice(args.sizes)), replace=False)
                               for p in np.unique(pid)])
        tx = texts[rows].tolist()
        G = mixed_clusters(tx, glm_l24[rows])                 # twin clusters (LM heads, LoRA)
        sz = np.bincount(G)[G]
        adapt = (sz >= 2) & (sz <= 18)
        Gs = sim.cluster_ids(tx)                              # the stylegram's own clusters
        szs = np.bincount(Gs)[Gs]
        srows = (szs >= 2) & (szs <= 18)
        Xc_s = sim.center(X[rows], Gs)

        lp = {key: plain[key][rows] for key in ('tokattn', 'tokattn_deb', 'lora', 'lora2')}
        lp_s, lp_sc = np.zeros((len(rows), K)), np.zeros((len(rows), K))
        for key in ('debv3l', 'electl'):
            lp[key] = np.zeros((len(rows), K))
        twin = {key: np.zeros((len(rows), K)) for key in ('q3_8b', 'glm9b', 'debv3l')}
        gate = np.zeros((len(rows), K))
        dc = np.zeros((len(rows), len(cl_idx)))
        fam_s = {f: np.zeros((len(rows), len(fidx))) for f, fidx, _ in (FAM if args.v559 else [])}
        heads = ('debv3l', 'electl') + (('q3_8b', 'glm9b') if args.screen else ())
        lm = {key: np.zeros((len(rows), K)) for key in heads[2:]}   # plain LM heads (weight 0 in the 0.556 notebook)
        std = {key: (E[key][rows] - E[key][rows].mean(0)) / (E[key][rows].std(0) + 1e-6) for key in heads}
        cen = {key: group_center(E[key][rows], G) for key in twin}
        for k in range(5):
            mk = fold[rows] == k
            m = M[k]
            lp_s[mk] = sim.svc_logp(m['svc'], X[rows][mk], K)
            lp_sc[mk] = sim.svc_logp(m['svc_c'], Xc_s[mk], K)
            for key in heads:   # test-set standardization, as fit_head
                (lp if key in lp else lm)[key][mk] = np.log(np.clip(m[key].predict_proba(std[key][mk]), 1e-9, 1))
            for key in twin:
                csc, clf = m[f'{key}_c']
                twin[key][mk] = np.log(np.clip(clf.predict_proba(csc.transform(cen[key][mk])), 1e-9, 1))
            dc[mk] = m['claude'].decision_function(XA_tr[rows][mk])
            for f in fam_s:
                dfn = m[f'fam_{f}'].decision_function(XB_tr[rows][mk])
                fam_s[f][mk] = np.stack([-dfn, dfn], 1) if dfn.ndim == 1 else dfn
            gsc, gclf = m['gate']
            gate[np.flatnonzero(mk)[:, None], gc_idx[None, :]] = np.log(np.clip(gclf.predict_proba(gsc.transform(E['glm9b'][rows][mk])), 1e-9, 1))
        # LoRA centered twin: centered features x the row's own fold readout
        Zc = group_center(lora_feats[rows], G)
        twin['lora'] = np.stack([softmax_log(Zc[i:i + 1] @ lora_ro[r]['center_coef'].T + lora_ro[r]['center_intercept'])[0]
                                 for i, r in enumerate(rows)])

        if args.v559:
            curly = np.array([any(c in t for c in ('\u2019', '\u201c', '\u201d')) for t in tx])
            em2 = np.array([t.count('\u2014') >= 2 for t in tx])
            cg = np.array([l.startswith(('claude', 'glm')) for l in labels])
            ge = np.array([l.startswith('gemini') for l in labels])
            shield_pen = -2.5 * (curly[:, None] * cg[None] + em2[:, None] * ge[None])
            # step 5 invariants (05_invariants.py: <= 1% in train and in prvsiyan's data, per model)
            apos = np.array([re.search(r"\w'\w", t) is not None for t in tx])
            em3 = np.array([t.count('\u2014') >= 3 for t in tx])
            sl = np.isin(labels, ['gpt-5.6-sol', 'gpt-5.6-luna'])
            g3 = np.isin(labels, ['gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.8-flash', 'qwen3.8-max'])
            pen_apos, pen_em3 = -2.5 * apos[:, None] * sl[None], -2.5 * em3[:, None] * g3[None]

        def blend(style_mode='replace', alpha=0.5, use_gate=False, lm_w=None, style_x=1.0, twin_x=1.0,
                  shield=False, fam_gates=False, adapt_claude=False, extra_pen=0.0):
            s = lp_s.copy()
            if style_mode == 'replace':
                s[srows] = lp_sc[srows]
            elif style_mode == 'half':   # same total weight, split between plain and centered
                s[srows] = 0.5 * lp_s[srows] + 0.5 * lp_sc[srows]
            L = style_x * W['stylegram'] * s + sum(W[key] * lp[key] for key in lp)
            if style_mode == 'add':
                L[srows] += W['stylegram'] * lp_sc[srows]
            for key, w in (lm_w or {}).items():
                L += w * lm[key]
            for key, w in TWIN_W.items():
                L[adapt] += twin_x * w * twin[key][adapt]
            P = np.exp(L - L.max(1, keepdims=True))
            P /= P.sum(1, keepdims=True)
            target = (1 - alpha) * P.sum(0) + alpha * np.full(K, len(P) / K)
            for _ in range(50):
                P *= (target / (P.sum(0) + 1e-12))[None, :]
                P /= P.sum(1, keepdims=True) + 1e-12
            if shield:   # the 0.559 notebook's post-Sinkhorn orthographic shield
                lpz = np.log(np.clip(P, 1e-300, 1)) + shield * shield_pen + extra_pen
                P = np.exp(lpz - lpz.max(1, keepdims=True))
                P /= P.sum(1, keepdims=True)
            pred = P.argmax(1)
            if use_gate:
                isg = np.isin(pred, gc_idx)
                new = gc_idx[(np.log(np.clip(P[:, gc_idx], 1e-12, 1)) + gate[:, gc_idx]).argmax(1)]
                pred = np.where(isg, new, pred)
            isc = np.isin(pred, cl_idx)   # within-Claude gate (every LB notebook has it)
            expert = CLAUDE_GATE_LAM * (fam_s['claude'] if fam_gates else dc)
            if adapt_claude:   # 0.559: shrink the gate when the blend is sure of its Claude pick
                rk = np.sort(P[:, cl_idx], 1)
                expert = expert * np.clip(0.25 + 0.75 * rk[:, -2] / np.maximum(rk[:, -1], 1e-12), 0.25, 1.0)[:, None]
            new = cl_idx[(np.log(np.clip(P[:, cl_idx], 1e-12, 1)) + expert).argmax(1)]
            pred = np.where(isc, new, pred)
            for f, fidx, _ in (FAM[1:] if fam_gates else []):   # 0.559: GPT, Gemini, GLM gates at lambda 0.25
                new = fidx[(np.log(np.clip(P[:, fidx], 1e-12, 1)) + 0.25 * fam_s[f]).argmax(1)]
                pred = np.where(np.isin(pred, fidx), new, pred)
            return pred

        variants = {'base': blend(), 'no_center': blend('none'), 'add': blend('add'),
                    'alpha025': blend(alpha=0.25), 'glm_gate': blend(use_gate=True)}
        if args.screen:
            variants.update({f'glm_w{w:g}': blend(lm_w={'glm9b': w}) for w in (0.1, 0.2, 0.4)})
            variants.update({'q3_w0.2': blend(lm_w={'q3_8b': 0.2}), 'both_w0.2': blend(lm_w={'glm9b': 0.2, 'q3_8b': 0.2})})
            variants.update({f'style_x{x:g}': blend(style_x=x) for x in (0.8, 1.25)})
            variants.update({f'twin_x{x:g}': blend(twin_x=x) for x in (0.67, 1.5)})
            variants['half'] = blend('half')
        if args.v559:
            variants.update({'shield': blend(shield=True), 'fam_gates': blend(fam_gates=True),
                             'adapt_claude': blend(adapt_claude=True),
                             'all559': blend(alpha=0.25, shield=True, fam_gates=True, adapt_claude=True)})
            v559 = dict(alpha=0.25, fam_gates=True, adapt_claude=True)
            variants.update({'559_shield2x': blend(shield=2, **v559),
                             '559_apos': blend(shield=True, extra_pen=pen_apos, **v559),
                             '559_em3': blend(shield=True, extra_pen=pen_em3, **v559),
                             '559_apos_em3': blend(shield=True, extra_pen=pen_apos + pen_em3, **v559)})
        for key, p in variants.items():
            res.setdefault(key, []).append(balanced_accuracy_score(y[rows], p))
        print(f'draw {d}: {len(rows)} passages, twin rows {adapt.sum()}, stylegram rows {srows.sum()} | '
              + ' '.join(f'{k} {v[-1]:.4f}' for k, v in res.items()), flush=True)

    lb = {'base': 0.556, 'no_center': 0.552, 'add': 0.550, 'alpha025': 0.556, 'glm_gate': 0.555}
    b = np.array(res[args.ref])
    rng = np.random.default_rng(0)
    print(f'\n{args.draws} draws, {args.sizes} passages per prompt (paired deltas vs {args.ref}; LB delta vs 0.556):')
    for key, s in res.items():
        dlt = np.array(s) - b
        pb = np.mean(rng.choice(dlt, (2000, len(dlt))).mean(1) > 0)   # bootstrap over draws
        print(f'  {key:10s} sim {np.mean(s):.4f}  delta {dlt.mean():+.4f} (wins {int((dlt > 0).sum())}/{len(s)}, '
              f'P(better) {pb:.2f})' + (f' | LB {lb[key]:.3f} delta {lb[key] - 0.556:+.3f}' if key in lb else ''))


if __name__ == '__main__':
    main()
