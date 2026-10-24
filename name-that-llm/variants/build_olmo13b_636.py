"""Build /home/user/work/push_ntl636_olmo13b: the 0.636 notebook (fleongg/ntl-part-two-complement copy) + our OLMo-13B member.

What: an exact copy of /home/user/work/push_ntl636 with two additions:
1. a credits/description markdown cell at the top;
2. after the part-two LoRA loop (Qwen3-14B, Qwen2.5-14B), one cell that writes lab/olmo13b_member.py verbatim and adds
   lora_olmo2_13b (aryanawasthiteykik/ntl-olmo13b-member) exactly as the loop adds a LoRA: weight W * log P(plain head),
   plus the centred twin at TWIN_W on the same topic clusters. It fails closed if the member is missing.
kernel-metadata.json gets a new private id, the member dataset and allenai/olmo-2/Transformers/1124-13b/1; the latest
environment, as the original.
Why: a new member from another model family, trained by us with prompt-grouped OOF (lab/09_train_olmo13b.py).
W = 0.2 / TWIN_W = 0.1 are the Qwen2.5-14B member's weights (a conservative choice until the OOF is in; see 09's merge).
Output: the push dir (not pushed). Usage: python build_olmo13b_636.py [--w 0.2] [--twin-w 0.1]
"""
import argparse
import json
import os

SRC, DST = '/home/user/work/push_ntl636', '/home/user/work/push_ntl636_olmo13b'
NB = 'ntl-part-two-complement.ipynb'
MEMBER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lab', 'olmo13b_member.py')

CREDITS = """**Variant** of our pinned copy of [fleongg/ntl-part-two-complement](https://www.kaggle.com/code/fleongg/ntl-part-two-complement) (public 0.636), code unchanged except one added member: our two-stage LoRA on **OLMo-2-13B** (`lora_olmo2_13b`), added to the part-two blend like its 14B LoRAs (weight __W__ on log probabilities, centred twin __TW__).

The member: [allenai/OLMo-2-1124-13B](https://www.kaggle.com/models/allenai/olmo-2/transformers/1124-13b) (Apache-2.0) by the Allen Institute for AI, first 27 of 40 layers, 4-bit NF4 base with an unmerged LoRA (rank 16), trained by us with Nishant Dahal's two-stage recipe: stage 1 on his public pool ([nishantdahal/ntl-handwriting-blend](https://www.kaggle.com/datasets/nishantdahal/ntl-handwriting-blend), CC BY-SA 4.0, whose source datasets are credited below), stage 2 on the competition's train. Checkpoint: aryanawasthiteykik/ntl-olmo13b-member (CC BY-SA 4.0).

Credit for everything else to fleongg, Nishant Dahal, and all authors credited below."""

MD = """## OLMo-2-13B LoRA (added)

Our member, scored with the code it was trained with (`olmo13b_member.py`, written below): the 4-bit NF4 base and the
LoRA kept unmerged, because the heads were trained on NF4 features. It joins the blend like the two LoRAs above."""

CODE = """from pathlib import Path
Path('/kaggle/working/olmo13b_member.py').write_text(__SRC__)
from olmo13b_member import OLMO_SPEC, olmo_features
OLMO13B = dict(w=__W__, twin_w=__TW__, bs=8)
ckpt = str(stage_checkpoint(os.path.dirname(find("lora_olmo2_13b/meta.json"))))
Z, heads = olmo_features(ckpt, find_base(OLMO_SPEC), test.text.tolist(), OLMO13B["bs"], LABELS)
vote("lora_olmo2_13b", softmax(Z @ heads["plain.weight"].T + heads["plain.bias"]), OLMO13B["w"])
if twin_rows.any():
    Zc = Z.copy()
    for g in np.unique(groups[twin_rows]):
        Zc[groups == g] -= Zc[groups == g].mean(0)
    centred = Zc[twin_rows] @ heads["centered.weight"].T + heads["centered.bias"]
    vote("lora_olmo2_13b twin", softmax(centred), OLMO13B["twin_w"], rows=twin_rows)
del Z
gc.collect()
torch.cuda.empty_cache()
assert "lora_olmo2_13b" in MEMBER_EVIDENCE, "OLMo-13B member required"
"""


def cell(kind, src):
    c = dict(cell_type=kind, metadata={}, source=src.splitlines(True))
    if kind == 'code':
        c.update(outputs=[], execution_count=None)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--w', type=float, default=0.2)
    ap.add_argument('--twin-w', type=float, default=0.1)
    a = ap.parse_args()
    nb = json.load(open(os.path.join(SRC, NB)))
    cells = nb['cells']
    loop = [i for i, c in enumerate(cells) if 'for ckpt_name, mc in LORAS.items():' in ''.join(c['source'])]
    assert len(loop) == 1 and 'LORAS = {' in ''.join(cells[11]['source'])
    sub = lambda s: s.replace('__W__', str(a.w)).replace('__TW__', str(a.twin_w))
    code = sub(CODE).replace('__SRC__', repr(open(MEMBER).read()))
    new = [cell('markdown', sub(CREDITS))] + cells[:loop[0] + 1] + [cell('markdown', MD), cell('code', code)] + \
        cells[loop[0] + 1:]
    nb['cells'] = new
    os.makedirs(DST, exist_ok=True)
    json.dump(nb, open(os.path.join(DST, NB), 'w'), indent=1)
    meta = json.load(open(os.path.join(SRC, 'kernel-metadata.json')))
    meta.update(id='aryanawasthiteykik/ntl-0-636-olmo13b', title='NTL 0.636 OLMo13B')
    meta['dataset_sources'] = meta['dataset_sources'] + ['aryanawasthiteykik/ntl-olmo13b-member']
    meta['model_sources'] = meta['model_sources'] + ['allenai/olmo-2/Transformers/1124-13b/1']
    json.dump(meta, open(os.path.join(DST, 'kernel-metadata.json'), 'w'), indent=1)
    print('wrote', DST, f'w={a.w} twin_w={a.twin_w}; cells {len(cells)} -> {len(new)}; member cell after cell {loop[0]}')


if __name__ == '__main__':
    main()
