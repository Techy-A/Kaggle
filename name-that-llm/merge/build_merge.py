"""Merge the two best public NTL notebooks that both extend lucasboesen/lora-x-adaptive-centering v3 (0.549):
  * lightningv08/0-551-lora-tabpfn-ensemble (0.551): v3 run as a script + TabPFN complement + punctuation rules
  * zbigniewkirchner/ntl-v6 (0.552): v3 + a third Qwen3-8B LoRA (lora3, Claude head) + within-Claude gates
The 0.551 notebook is the shell. Its lucas_reference.py (exact v3) is replaced by the 0.552 main code (v3 + lora3 +
Claude models). The 0.552 code saves its Claude-gate scores, and the Claude gate is re-applied as the very last step,
after the TabPFN mix and punctuation rules (otherwise the finishing step would overwrite it).
usage: python build_merge.py <lv_551_dir> <zb_v6_dir> <out_dir>"""
import json, os, sys

lv_dir, zb_dir, out = sys.argv[1:4]
lv = json.load(open(os.path.join(lv_dir, '0-551-lora-tabpfn-ensemble.ipynb')))
zb = json.load(open(os.path.join(zb_dir, 'ntl-v6.ipynb')))
src = lambda c: ''.join(c['source'])
zb_code = src(zb['cells'][1])
loop = 'for key, mt, hid, layers, C, smoke_id, pool in LLM_HEADS:'
assert zb_code.count(loop) == 1 and zb_code.count('pred = P.argmax(1)') == 1
gate_old = 'if DC_TE is not None:\n    _isc = np.isin(pred, CLAUDE_IDX)'
assert zb_code.count(gate_old) == 1
save = ("import os as _mos\n"
        "if _mos.environ.get('NTL_TABULAR_TEMP'):   # MERGE: hand the Claude-gate scores to the final step\n"
        "    np.savez(_mos.path.join(_mos.environ['NTL_TABULAR_TEMP'], 'claude_gate.npz'),\n"
        "             idx=np.asarray(CLAUDE_IDX), dc=(DC_TE if DC_TE is not None else np.zeros((0, 0))),\n"
        "             lc=(LC_TE if LC_TE is not None else np.zeros((0, 0))), lam=np.array([CLAUDE_GATE_LAM, CLAUDE_LORA_LAM]))\n")
zb_code = zb_code.replace(gate_old, save + gate_old)

cells = lv['cells']
c6 = src(cells[6]); first = c6.split('\n', 1)[0]
assert first.startswith('%%writefile') and first.endswith('lucas_reference.py')
cells[6]['source'] = (first + '\n' + zb_code).splitlines(True)
# run_baseline: the reference is no longer the exact v3 file, so record its hash instead of asserting it
c10 = src(cells[10])
old = "assert hashlib.sha256(raw).hexdigest()==config['reference_source_sha256']\n"
assert c10.count(old) == 1
cells[10]['source'] = c10.replace(old, "print('MERGE reference = zbigniewkirchner/ntl-v6 main code, sha256', hashlib.sha256(raw).hexdigest(), flush=True)\n").splitlines(True)
# finish_prediction: re-apply the Claude gate after the TabPFN mix and punctuation rules
c18 = src(cells[18])
old = "P=apply_post_balance(mixed,texts,labels)\nsubmission=pd.DataFrame(dict(id=test.id,model=labels[P.argmax(1)]))\n"
assert c18.count(old) == 1
new = ("P=apply_post_balance(mixed,texts,labels)\n"
       "pred=P.argmax(1)\n"
       "_g=cache/'claude_gate.npz'\n"
       "if _g.is_file():   # MERGE: zbigniewkirchner's within-Claude gate, applied last\n"
       "    with np.load(_g,allow_pickle=False) as G:\n"
       "        idx,dc,lc,lam=G['idx'],G['dc'],G['lc'],G['lam']\n"
       "    if dc.size:\n"
       "        assert dc.shape==(len(P),len(idx))\n"
       "        isc=np.isin(pred,idx)\n"
       "        s=np.log(np.clip(P[:,idx],1e-12,1))+lam[0]*dc\n"
       "        if lc.size:\n"
       "            assert lc.shape==dc.shape; s=s+lam[1]*lc\n"
       "        new=idx[s.argmax(1)]\n"
       "        print('MERGE claude gate changed',int((new[isc]!=pred[isc]).sum()),'of',int(isc.sum()),'Claude predictions',flush=True)\n"
       "        pred[isc]=new[isc]\n"
       "submission=pd.DataFrame(dict(id=test.id,model=labels[pred]))\n")
cells[18]['source'] = c18.replace(old, new).splitlines(True)
c18 = src(cells[18]); c18 = c18.replace("prediction_changes_vs_v3=int((P.argmax(1)!=base.argmax(1)).sum()),", "prediction_changes_vs_v3=int((pred!=base.argmax(1)).sum()),")
cells[18]['source'] = c18.splitlines(True)
title = {'cell_type': 'markdown', 'metadata': {}, 'source': [
    '# NTL merge: 0.551 TabPFN/punctuation + 0.552 lora3/Claude gates\n', '\n',
    'Combines two public notebooks that both extend [lucasboesen/lora-x-adaptive-centering](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering) v3 (0.549): '
    '[lightningv08/0-551-lora-tabpfn-ensemble](https://www.kaggle.com/code/lightningv08/0-551-lora-tabpfn-ensemble) (TabPFN complement, punctuation rules) and '
    '[zbigniewkirchner/ntl-v6](https://www.kaggle.com/code/zbigniewkirchner/ntl-v6) (third LoRA with a Claude head, within-Claude gates). '
    'The Claude gate is applied last. All credit to the original authors (Apache 2.0); please upvote them.\n']}
lv['cells'] = [title] + cells
md_lv = json.load(open(os.path.join(lv_dir, 'kernel-metadata.json'))); md_zb = json.load(open(os.path.join(zb_dir, 'kernel-metadata.json')))
assert md_lv['docker_image'] == md_zb['docker_image'] and md_lv['kernel_sources'] == md_zb['kernel_sources'] and md_lv['model_sources'] == md_zb['model_sources']
ds = list(dict.fromkeys(md_lv['dataset_sources'] + md_zb['dataset_sources']))
md = dict(id='aryanawasthiteykik/ntl-merge-tabpfn-claude-gates', title='NTL merge TabPFN + Claude gates', code_file='notebook.ipynb',
          language='python', kernel_type='notebook', is_private=True, enable_gpu=True, enable_tpu=False,
          machine_shape=md_lv['machine_shape'], enable_internet=False, docker_image=md_lv['docker_image'],
          docker_image_pinning_type='original', competition_sources=md_lv['competition_sources'], dataset_sources=ds,
          kernel_sources=md_lv['kernel_sources'], model_sources=md_lv['model_sources'])
os.makedirs(out, exist_ok=True)
json.dump(lv, open(os.path.join(out, 'notebook.ipynb'), 'w'), indent=1)
json.dump(md, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=1)
print('wrote', out, len(lv['cells']), 'cells | datasets', ds)
