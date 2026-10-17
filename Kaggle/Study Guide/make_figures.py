"""Figures for the Study Guide (static PNGs for Obsidian). Re-run: python make_figures.py"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')
os.makedirs(OUT, exist_ok=True)
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e6e5e1'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
                     'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'text.color': INK, 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'axes.axisbelow': True, 'grid.color': GRID, 'grid.linewidth': 0.8})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches='tight')
    plt.close(fig)


def box(ax, x, y, w, h, text, color=BLUE, fc=None, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.04',
                                fc=fc or SURF, ec=color, lw=2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', color=INK2, lw=1.6))


def blank(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')


# 1. random vs grouped CV
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
rng = np.random.default_rng(0)
for ax, title, grouped in [(axes[0], 'Random split: same prompt in train AND validation', False),
                           (axes[1], 'Grouped split (by prompt): validation prompts unseen', True)]:
    for p in range(8):
        for m in range(6):
            val = (p in (2, 5)) if grouped else rng.random() < 0.25
            ax.add_patch(Rectangle((m, 7 - p), 0.9, 0.9, fc=ORANGE if val else BLUE, ec=SURF, lw=2))
    ax.set_xlim(-0.2, 6); ax.set_ylim(-0.2, 8); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(title, fontsize=10.5, color=INK); ax.set_xlabel('6 passages (one per model)'); ax.set_ylabel('8 prompts')
fig.text(0.5, -0.04, 'blue = train   orange = validation', ha='center', color=INK2)
save(fig, 'grouped_cv.png')

# 2. leakage diagram (our np bench)
fig, ax = plt.subplots(figsize=(10, 4)); blank(ax)
box(ax, 0.2, 3.6, 2.8, 1.4, 'Molecule X\nspectra in RIKEN /\nGNPS (in train)', ORANGE)
box(ax, 3.8, 3.6, 2.6, 1.4, 'Model trained\non train.parquet\n(memorises X)', BLUE)
box(ax, 7.1, 3.6, 2.7, 1.4, 'Bench: hide X,\nask model to find X\n-> "easy"!', ORANGE)
arrow(ax, 3.0, 4.3, 3.8, 4.3); arrow(ax, 6.4, 4.3, 7.1, 4.3)
box(ax, 0.2, 0.6, 2.8, 1.4, 'Molecule U\nabsent from train\n(Zenodo extra)', AQUA)
box(ax, 3.8, 0.6, 2.6, 1.4, 'Same model\n(never saw U)', BLUE)
box(ax, 7.1, 0.6, 2.7, 1.4, 'Unseen bench:\nhonest score', AQUA)
arrow(ax, 3.0, 1.3, 3.8, 1.3); arrow(ax, 6.4, 1.3, 7.1, 1.3)
ax.text(5, 5.6, 'Leakage: the validation answer is already inside the model', ha='center', fontsize=12, color=INK)
save(fig, 'leakage.png')

# 3. winner's curse
rng = np.random.default_rng(1)
true = 0.451; n_sub = 30; trials = 4000
picked = [true + rng.normal(0, 0.003, n_sub).max() for _ in range(trials)]
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.hist(picked, bins=40, color=BLUE, edgecolor=SURF, linewidth=1.5)
ax.axvline(true, color=INK2, lw=2, ls='--')
ax.text(true, ax.get_ylim()[1] * 0.92, '  true quality 0.451', color=INK2, fontsize=10)
ax.set_xlabel('public score of the BEST of 30 equally good submissions (noise sd 0.003)'); ax.set_ylabel('count')
ax.set_title("Winner's curse: picking the max of noisy scores overestimates the truth", fontsize=11, color=INK)
save(fig, 'winners_curse.png')

# 4. our real noise band
fig, ax = plt.subplots(figsize=(9, 3.6))
names = ['V22', 'V22 (Kino copy)', 'depth 7', 'depth 7 (Kino rerun)', 'E2R3', 'rel100']
vals = [0.451, 0.451, 0.451, 0.447, 0.450, 0.445]
ax.axhspan(0.447, 0.455, color=GRID, alpha=0.7, lw=0)
ax.text(5.45, 0.4545, 'about ±0.004 = noise', ha='right', color=INK2, fontsize=9)
ax.bar(range(len(vals)), np.array(vals) - 0.44, bottom=0.44, color=[BLUE, BLUE, AQUA, AQUA, ORANGE, ORANGE], width=0.6, edgecolor=SURF, lw=2)
for i, v in enumerate(vals): ax.text(i, v + 0.0004, f'{v:.3f}', ha='center', fontsize=9, color=INK)
ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, fontsize=9); ax.set_ylim(0.44, 0.457); ax.set_ylabel('public LB (CASMI)')
ax.set_title('Real example: same code scored 0.451 and 0.447', fontsize=11, color=INK)
save(fig, 'noise_band.png')

# 5. boosting train/valid curve
it = np.arange(1, 501)
train = 0.70 * np.exp(-it / 120) + 0.20
valid = 0.70 * np.exp(-it / 120) + 0.27 + 0.00025 * np.maximum(0, it - 180)
best = int(np.argmin(valid))
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(it, train, color=BLUE, lw=2, label='train loss'); ax.plot(it, valid, color=ORANGE, lw=2, label='validation loss')
ax.axvline(it[best], color=INK2, ls='--', lw=1.5); ax.text(it[best] + 5, 0.82, f'early stop ~{it[best]} rounds', color=INK2, fontsize=9)
ax.set_xlabel('boosting rounds'); ax.set_ylabel('loss'); ax.legend(frameon=False)
ax.set_title('Gradient boosting: stop where validation loss bottoms out', fontsize=11, color=INK)
save(fig, 'boosting_early_stop.png')

# 6. LoRA
fig, ax = plt.subplots(figsize=(9, 3.6)); blank(ax)
box(ax, 0.3, 1.5, 2.8, 3, 'Frozen weight W\n(d x d)\n\nbillions of params\nNOT trained', MUTED, fs=10)
ax.text(3.55, 3, '+', fontsize=26, ha='center', va='center')
box(ax, 4.1, 1.5, 0.8, 3, 'A\nd x r', ORANGE)
ax.text(5.25, 3, 'x', fontsize=18, ha='center', va='center')
box(ax, 5.6, 2.6, 3, 0.8, 'B   r x d', ORANGE)
ax.text(5, 0.6, 'r = 8..64  ->  ~0.1-1% of the parameters are trained', ha='center', fontsize=10, color=INK2)
ax.text(5, 5.4, 'LoRA: W_new = W + A·B (only A and B learn)', ha='center', fontsize=12)
save(fig, 'lora.png')

# 7. two-stage retrieval with our gen20 numbers
fig, ax = plt.subplots(figsize=(10, 3.6)); blank(ax)
box(ax, 0.2, 2.2, 2.4, 1.6, 'Spectrum\n(query)', MUTED)
box(ax, 3.1, 2.2, 2.8, 1.6, 'Stage 1: candidates\n(lookup + generation)\nrecall matters', BLUE)
box(ax, 6.4, 2.2, 1.7, 1.6, 'Stage 2:\nre-rank', AQUA)
box(ax, 8.5, 2.2, 1.3, 1.6, 'top 25', MUTED)
arrow(ax, 2.6, 3, 3.1, 3); arrow(ax, 5.9, 3, 6.4, 3); arrow(ax, 8.1, 3, 8.5, 3)
ax.text(4.5, 1.3, 'gen20: novel answers generated 46 -> 69 of 130', ha='center', fontsize=10, color=INK2)
ax.text(5, 5.2, 'If the answer is not a candidate, no re-ranker can save it', ha='center', fontsize=12)
save(fig, 'two_stage_retrieval.png')

# 8. reciprocal rank
fig, ax = plt.subplots(figsize=(9, 3.4))
r = np.arange(1, 11)
ax.bar(r, 1 / r, color=BLUE, edgecolor=SURF, lw=2, width=0.7)
for x in [1, 2, 3, 5, 10]: ax.text(x, 1 / x + 0.02, f'{1/x:.2f}', ha='center', fontsize=9, color=INK)
ax.set_xticks(r); ax.set_xlabel('rank of the first correct answer'); ax.set_ylabel('score (1/rank)')
ax.set_title('MRR: rank 1 -> 2 costs 0.5 of a molecule; rank 10 -> 11 costs almost nothing', fontsize=11, color=INK)
save(fig, 'mrr.png')

# 9. calibration
p = np.linspace(0, 1, 11)
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.plot([0, 1], [0, 1], color=MUTED, ls='--', lw=1.5, label='perfect')
ax.plot(p, p ** 1.8, color=ORANGE, lw=2, marker='o', ms=6, label='over-confident model')
ax.plot(p, np.clip(p + 0.02 * np.sin(p * 6), 0, 1), color=BLUE, lw=2, marker='o', ms=6, label='after temperature scaling')
ax.set_xlabel('predicted probability'); ax.set_ylabel('actual frequency'); ax.legend(frameon=False, fontsize=9)
ax.set_title('Reliability diagram', fontsize=11, color=INK)
save(fig, 'calibration.png')

# 10. label shift / Sinkhorn balancing
k = np.arange(6)
raw = np.array([0.30, 0.22, 0.18, 0.13, 0.10, 0.07])
alpha = 0.5; bal = (1 - alpha) * raw + alpha / 6
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.bar(k - 0.18, raw, width=0.34, color=ORANGE, edgecolor=SURF, lw=2, label='predicted class mix (raw)')
ax.bar(k + 0.18, bal, width=0.34, color=BLUE, edgecolor=SURF, lw=2, label='after Sinkhorn balancing (alpha 0.5)')
ax.axhline(1 / 6, color=INK2, ls='--', lw=1.2); ax.text(5.45, 1 / 6 + 0.006, 'uniform', ha='right', fontsize=9, color=INK2)
ax.set_xticks(k); ax.set_xticklabels([f'model {i+1}' for i in k]); ax.set_ylabel('share of predictions'); ax.legend(frameon=False, fontsize=9)
ax.set_title('Label-shift correction: pull predicted class counts toward the expected mix', fontsize=11, color=INK)
save(fig, 'sinkhorn.png')

# 11. ensembling: reciprocal-rank fusion
fig, ax = plt.subplots(figsize=(10, 3.8)); blank(ax)
box(ax, 0.3, 3.4, 2.4, 1.9, 'Engine 1\n1. A\n2. B\n3. C', BLUE, fs=10)
box(ax, 0.3, 0.6, 2.4, 1.9, 'Engine 2\n1. B\n2. D\n3. A', ORANGE, fs=10)
box(ax, 4.0, 1.7, 3.2, 2.4, 'RRF score\n= sum 1/(k + rank)\nA: 1/4+1/6\nB: 1/5+1/4', AQUA, fs=10)
box(ax, 8.0, 1.9, 1.8, 2.0, 'Fused\n1. B\n2. A\n3. D', MUTED, fs=10)
arrow(ax, 2.7, 4.3, 4.0, 3.4); arrow(ax, 2.7, 1.5, 4.0, 2.4); arrow(ax, 7.2, 2.9, 8.0, 2.9)
ax.text(5, 5.6, 'Ensembling ranked lists: items both engines like rise to the top (k = 3)', ha='center', fontsize=12)
save(fig, 'rrf.png')

# 12. tautomers (RDKit)
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.MolStandardize import rdMolStandardize
    keto, enol = Chem.MolFromSmiles('CC(=O)CC(C)=O'), Chem.MolFromSmiles('CC(O)=CC(C)=O')
    t = rdMolStandardize.TautomerEnumerator()
    ka = Chem.MolToInchiKey(t.Canonicalize(keto))[:14]; kb = Chem.MolToInchiKey(t.Canonicalize(enol))[:14]
    raw_a, raw_b = Chem.MolToInchiKey(keto)[:14], Chem.MolToInchiKey(enol)[:14]
    img = Draw.MolsToGridImage([keto, enol], molsPerRow=2, subImgSize=(360, 260),
                               legends=[f'keto  raw {raw_a}\nmetric {ka}', f'enol  raw {raw_b}\nmetric {kb}'])
    img.save(os.path.join(OUT, 'tautomers.png'))
except Exception as e:
    print('rdkit figure skipped', e)
print('figures in', OUT, sorted(os.listdir(OUT)))
