"""Add a GLM+Claude gate on frozen GLM-4-9B features to an NTL notebook (one change).

The lab simulation (name-that-llm/lab/03_llm_gate.py: held-out prompts, 2-3 passages per prompt) found GLM and Claude
models blurring into each other; a 7-way logistic gate on GLM-4-9B L8+L3 features (StandardScaler + LR, trained on the GLM
and Claude rows only) re-deciding every GLM/Claude prediction added +0.035 balanced accuracy over the stylegram (5/5).
The notebook already computes these features for its glm9b head, so the gate costs no GPU. It runs on the balanced
posterior before the existing within-Claude gate, which still decides WHICH Claude afterwards.

usage: python build_glm_claude_gate.py <base push dir> <out dir> <kernel slug> <title>
"""
import json
import os
import sys

base, out, slug, title = sys.argv[1:5]
nb = json.load(open(os.path.join(base, 'notebook.ipynb')))
code = [c for c in nb['cells'] if c['cell_type'] == 'code']
assert len(code) == 1
src = ''.join(code[0]['source'])


def rep(a, b):
    global src
    assert src.count(a) == 1, a[:80]
    src = src.replace(a, b)


rep("""def fit_head(key, Etr, Ete, C):
""", """GC_GATE_LAM = 1.0   # GLM+Claude gate on glm9b features (lab sim, 2-3 passages/prompt: +0.035 over the stylegram, 5/5)
GC_IDX = np.array([i for i, l in enumerate(LABELS) if l.startswith(('glm', 'claude'))])
GC_TE = None


def fit_head(key, Etr, Ete, C):
""")
rep("""    sc = StandardScaler().fit(Etr)
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(Etr), y)
""", """    sc = StandardScaler().fit(Etr)
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(Etr), y)
    if key == 'glm9b':   # 7-way GLM+Claude gate on the same features, trained on those rows only (train scaling, as in the sim)
        global GC_TE
        _m = np.isin(y, GC_IDX)
        _gsc = StandardScaler().fit(Etr[_m])
        _g = LogisticRegression(C=C, max_iter=5000).fit(_gsc.transform(Etr[_m]), y[_m])
        assert list(_g.classes_) == list(GC_IDX)
        GC_TE = np.log(np.clip(_g.predict_proba(_gsc.transform(Ete)), 1e-9, 1))
        print('glm+claude gate trained on', int(_m.sum()), 'rows', flush=True)
""")
rep("""pred = P.argmax(1)
if DC_TE is not None:
""", """pred = P.argmax(1)
if GC_TE is not None:   # GLM+Claude gate first; the within-Claude gate below still picks WHICH Claude
    _isg = np.isin(pred, GC_IDX)
    _new = GC_IDX[(np.log(np.clip(P[:, GC_IDX], 1e-12, 1)) + GC_GATE_LAM * GC_TE).argmax(1)]
    print('glm+claude gate changed', int((_new[_isg] != pred[_isg]).sum()), 'of', int(_isg.sum()), 'GLM/Claude predictions', flush=True)
    pred[_isg] = _new[_isg]
if DC_TE is not None:
""")
compile(src, 'nb', 'exec')
code[0]['source'] = src.splitlines(True)
nb['cells'][0]['source'] = nb['cells'][0]['source'] + [
    "\n", "**+ GLM+Claude gate.** When the balanced posterior predicts a GLM or Claude model, a 7-way logistic gate on the "
    "frozen GLM-4-9B L8+L3 features (already computed for the glm9b head) re-decides among those 7, before the existing "
    "within-Claude gate. Lab simulation with 2-3 passages per prompt: +0.035 balanced accuracy over the stylegram (5/5 draws).\n"]
os.makedirs(out, exist_ok=True)
json.dump(nb, open(os.path.join(out, 'notebook.ipynb'), 'w'), indent=1)
m = json.load(open(os.path.join(base, 'kernel-metadata.json')))
m.update(id=f'aryanawasthiteykik/{slug}', title=title)
json.dump(m, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=1)
print('wrote', out)
