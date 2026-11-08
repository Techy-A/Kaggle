"""22: are Open Pool rows labelled gemini-3.7-flash really 3.8-flash? (label-noise audit by source)

What: a 3.7-vs-3.8 classifier (TF-IDF char 2-5 + word 1-2, logistic regression) trained only on the official train
passages of the two labels (150 + 150, true labels), validated by prompt-grouped 5-fold CV; then applied to Open Pool v15
rows labelled 3.7 or 3.8 (not train-prompt regenerations), reporting per source_dataset the mean P(3.8) and the share
of rows called 3.8. A 3.7 source scoring like the 3.8 reference is suspect (e.g. generated through an endpoint that
aliases 3.7 to 3.8, as AI Studio does: lab/20 pilot, 8 Nov).
Output: artifacts/22_gemini37_label_audit.json. Usage: python 22_gemini37_label_audit.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).parent
t = pd.read_csv(HERE / 'data/train.csv'); t = t[t.model.isin(['gemini-3.7-flash', 'gemini-3.8-flash'])].reset_index(drop=True)
y = (t.model == 'gemini-3.8-flash').astype(int).values
def feats(fit_texts):
    c = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True).fit(fit_texts)
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(fit_texts)
    return lambda x: hstack([c.transform(x), w.transform(x)]).tocsr()
oof = np.zeros(len(t))
for tr, te in GroupKFold(5).split(t, y, t.prompt_id):
    f = feats(t.text.iloc[tr]); m = LogisticRegression(C=4, max_iter=3000).fit(f(t.text.iloc[tr]), y[tr]); oof[te] = m.predict_proba(f(t.text.iloc[te]))[:, 1]
auc = roc_auc_score(y, oof); acc = float(((oof > 0.5) == y).mean())
print(f'train 3.7-vs-3.8, prompt-grouped CV: AUC {auc:.3f}, acc {acc:.3f}  (n 150 + 150)')
f = feats(t.text); m = LogisticRegression(C=4, max_iter=3000).fit(f(t.text), y)
p = pd.read_parquet('/home/user/work/ds_pool_v15/ntl_open_pool.parquet', columns=['text', 'model', 'source_dataset', 'flag_train_prompt_regen', 'flag_kb_proxy'])
p = p[p.model.isin(['gemini-3.7-flash', 'gemini-3.8-flash']) & ~p.flag_train_prompt_regen.astype(bool)].copy()
p['p38'] = m.predict_proba(f(p.text))[:, 1]
rep = (p.groupby(['model', 'source_dataset']).p38.agg(n='size', mean_p38='mean', share_called_38=lambda s: float((s > 0.5).mean()))
       .reset_index().sort_values(['model', 'n'], ascending=[True, False]))
print(rep[rep.n >= 30].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
ref37 = float(oof[y == 0].mean()); ref38 = float(oof[y == 1].mean())
print(f'reference (train OOF): true 3.7 mean P(3.8) {ref37:.3f}; true 3.8 mean P(3.8) {ref38:.3f}')
json.dump(dict(cv_auc=auc, cv_acc=acc, ref_true37=ref37, ref_true38=ref38, by_source=rep.to_dict('records')),
          open(HERE / 'artifacts/22_gemini37_label_audit.json', 'w'), indent=1, default=float)
