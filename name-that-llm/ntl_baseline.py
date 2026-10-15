"""Name That LLM: TF-IDF (char + word) logistic regression + prompt-group assignment.

Every prompt in train.csv was answered exactly once by each of the 18 models. Test prompts are unseen, but test
passages can be grouped back into prompts by topic (content-word TF-IDF + average-linkage clustering, ARI 0.91 on
train). Within a group each model is used at most once (Hungarian assignment on log-probabilities).
Prompt-grouped 5-fold CV balanced accuracy: argmax 0.342 -> assignment 0.433.
"""
import glob

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import hstack
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

N_MODELS = 18


def find(name):
    hits = sorted(glob.glob(f'/kaggle/input/**/{name}', recursive=True), key=len)
    return hits[0] if hits else name


def style_features(train_text, test_text):
    vc = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 5), sublinear_tf=True, min_df=2, max_features=300000)
    vw = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, lowercase=False,
                         token_pattern=r"(?u)\b\w+\b|[^\w\s]")
    Xa = hstack([vc.fit_transform(train_text), vw.fit_transform(train_text)]).tocsr()
    Xb = hstack([vc.transform(test_text), vw.transform(test_text)]).tocsr()
    return Xa, Xb


def prompt_groups(texts):
    """Cluster passages into (approximately) their prompts by content words."""
    n = len(texts)
    if n < 2 * N_MODELS:
        return np.zeros(n, int)
    X = TfidfVectorizer(stop_words='english', sublinear_tf=True, min_df=2).fit_transform(texts)
    S = (X @ X.T).toarray()
    k = max(1, int(round(n / N_MODELS)))
    return AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average').fit_predict(1 - S)


def assign(P, groups, labels):
    pred = labels[P.argmax(1)].astype(object)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        k = int(np.ceil(len(idx) / len(labels)))
        C = -np.log(np.repeat(P[idx], k, axis=1) + 1e-9)
        r, c = linear_sum_assignment(C)
        pred[idx[r]] = labels[c // k]
    return pred


def main():
    train = pd.read_csv(find('train.csv'))
    test = pd.read_csv(find('test.csv'))
    Xa, Xb = style_features(train.text, test.text)
    model = LogisticRegression(C=10, max_iter=3000, class_weight='balanced').fit(Xa, train.model)
    P = model.predict_proba(Xb)
    labels = model.classes_
    pred = assign(P, prompt_groups(test.text.tolist()), labels)
    sub = pd.DataFrame({'id': test.id, 'model': pred})
    sub.to_parquet('submission.parquet', index=False)
    print(sub.shape, sub.model.value_counts().to_dict())


if __name__ == '__main__':
    main()
