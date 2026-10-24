"""Build /home/user/work/push_ntl639_olmo13b: the 0.639 notebook (kirilltym's, our pinned copy) + our OLMo-13B member.

What: an exact copy of /home/user/work/push_ntl639 with two additions, reusing build_olmo13b_636.py's text:
1. a credits/description markdown cell at the top (the copy's own "Exact copy" cell stays below it);
2. right after the LoRA loop (cell 19: Qwen3-14B, Qwen2.5-14B, the way nishantdahal's 14B members join), one cell that
   writes lab/olmo13b_member.py verbatim and adds lora_olmo2_13b (aryanawasthiteykik/ntl-olmo13b-member): weight W *
   log P(plain head) plus the centred twin at TWIN_W on the same topic clusters; fails closed if the member is missing.
   Guarded by `if not PUBLIC_PREVIEW:` like every cell of the base (the 3-row public fixture loads no weights).
kernel-metadata.json gets a new private id, the member dataset and allenai/olmo-2/Transformers/1124-13b/1; the latest
environment, as the base. Output: the push dir (not pushed). Usage: python build_olmo13b_639.py [--w 0.2] [--twin-w 0.1]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_olmo13b_636 as b6  # noqa: E402

SRC, DST = '/home/user/work/push_ntl639', '/home/user/work/push_ntl639_olmo13b'
NB = '0-639-teach-more-keep-the-signature.ipynb'
CREDITS = b6.CREDITS.replace('our pinned copy of [fleongg/ntl-part-two-complement](https://www.kaggle.com/code/fleongg/'
                             'ntl-part-two-complement) (public 0.636)', 'our pinned copy of [kirilltym/0-639-teach-more-'
                             'keep-the-signature](https://www.kaggle.com/code/kirilltym/0-639-teach-more-keep-the-signature) '
                             '(public 0.639)').replace('part-two blend like its 14B LoRAs', 'blend like its 14B LoRAs').replace(
    'Credit for everything else to fleongg,', 'Credit for everything else to kirilltym, fleongg,')
assert 'kirilltym' in CREDITS and 'fleongg/ntl-part' not in CREDITS.split('The member')[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--w', type=float, default=0.2)
    ap.add_argument('--twin-w', type=float, default=0.1)
    a = ap.parse_args()
    nb = json.load(open(os.path.join(SRC, NB)))
    cells = nb['cells']
    loop = [i for i, c in enumerate(cells) if 'for ckpt_name, mc in LORAS.items():' in ''.join(c['source'])]
    assert len(loop) == 1 and 'LORAS = {' in ''.join(cells[12]['source'])
    sub = lambda s: s.replace('__W__', str(a.w)).replace('__TW__', str(a.twin_w))
    code = sub(b6.CODE).replace('__SRC__', repr(open(b6.MEMBER).read()))
    code = 'if not PUBLIC_PREVIEW:\n' + ''.join('    ' + l if l.strip() else l for l in code.splitlines(True))
    new = [b6.cell('markdown', sub(CREDITS))] + cells[:loop[0] + 1] + [b6.cell('markdown', b6.MD), b6.cell('code', code)] + \
        cells[loop[0] + 1:]
    nb['cells'] = new
    os.makedirs(DST, exist_ok=True)
    json.dump(nb, open(os.path.join(DST, NB), 'w'), indent=1)
    meta = json.load(open(os.path.join(SRC, 'kernel-metadata.json')))
    meta.update(id='aryanawasthiteykik/ntl-0-639-olmo13b', title='NTL 0.639 OLMo13B')
    meta['dataset_sources'] = meta['dataset_sources'] + ['aryanawasthiteykik/ntl-olmo13b-member']
    meta['model_sources'] = meta['model_sources'] + ['allenai/olmo-2/Transformers/1124-13b/1']
    json.dump(meta, open(os.path.join(DST, 'kernel-metadata.json'), 'w'), indent=1)
    print('wrote', DST, f'w={a.w} twin_w={a.twin_w}; cells {len(cells)} -> {len(new)}; member cell after cell {loop[0]}')


if __name__ == '__main__':
    main()
