"""Candidate-level PubChem promotion gate (sanpier's 0.465 fix) ported onto our 0.464 copy (dmitriigluzdov v20/v24).

Why: in the 0.464 merge cell the rank-1 promotion of a PubChem proposal is decided on channel-level values: p['S'] and
p['top_pop'] belong to the S-leader of the worker's list, and rel uses fz_top, the best raw fingerprint score of any
non-pool structure. After the ICEBERG/GLACIER re-ordering the candidate actually promoted, p['pc'][0], can be a
different structure, so it can be promoted on another candidate's evidence (or blocked by it). sanpier's
casmi-26-the-data-behind-spectra-to-structures v2 (public 0.465, a fork of dmitriigluzdov v16 at 0.464) keys the
evidence by the retrieval score key and gates on the promoted candidate's own S, pop and raw margin. Thresholds stay
S > 6 with pop >= 5, OR margin > 150; interleaving slots keep the channel-level rel. Nothing else changes.

What, three edits, copied from the embedded source of sanpier/candidate-aware-pubchem-promotion-0-465
(combined_code/frozen_0465.ipynb, sha256 aa1a43cd...):
  1. cell 49, V17_PATCH string: the PubChem worker stores promotion_evidence[key] = {smiles, raw_fz, S, pop} for
     every listed candidate and returns it in diag;
  2. cell 49, RUNNER string: pc_lists.json carries promotion_evidence per molecule;
  3. cell 58 (merge): candidate_promotion_evidence(p) and the candidate-level gate, plus pubchem_promotion_audit.csv.
The v20 five-fold FPNet ensemble and the sixth PubChem slot (rank 24) of our base are kept.

usage: python port_candgate.py <0.464 push dir> <out dir> <kernel slug> <title>
"""
import ast
import glob
import json
import os
import sys

PATCH_OLD = ("        out.append(smis[idx[j]]); out_s.append(float(sc[j])); out_k.append(k); "
             "out_p.append(float(pop[idx[j]])); out_sc.append(float(sc[j]))\n")
PATCH_NEW = PATCH_OLD + ("        promotion_evidence[k] = dict(smiles=smis[idx[j]], raw_fz=float(fz[j]),\n"
                         "                                     S=float(sc[j]), pop=float(pop[idx[j]]))\n")
LIST_OLD = "    out, out_s, out_k, out_p, out_sc, seen = [], [], [], [], [], set()\n"
LISTED_OLD = "    diag['n_listed'] = len(out)\n"
RUNNER_OLD = "top_pop=d.get('top_pop', 0.0), **extra[mid])"
RUNNER_NEW = "top_pop=d.get('top_pop', 0.0), promotion_evidence=d.get('promotion_evidence', {}), **extra[mid])"

GATE_FN = '''def candidate_promotion_evidence(p):
    """Look up the current leader by its immutable retrieval identity."""
    key = p['pc_keys'][0]
    evidence = p.get('promotion_evidence', {}).get(key)
    assert evidence is not None, 'Missing candidate promotion evidence; rerun PubChem: ' + str(key)
    assert evidence['smiles'] == p['pc'][0], 'PubChem identity/representative mismatch'
    S, pop, raw = (float(evidence[x]) for x in ('S', 'pop', 'raw_fz'))
    assert all(np.isfinite(v) for v in (S, pop, raw)), 'Nonfinite promotion evidence'
    # Preserve the baseline's no-pool-evidence convention.
    rel = raw - p['best_pool_fz'] if np.isfinite(p['best_pool_fz']) else 1e9
    return S, pop, rel


promotion_audit = []
'''
LOOP_OLD = "rows, stats = [], dict(untouched=0, gentle=0, aggressive=0, no_pc=0, promoted=0)\n"
GATE_OLD = """        S = p.get('S'); pop_top = float(p.get('top_pop') or 0.0)
        # Skomuro LB 0.403 Rule: Promote on Popularity OR Structure-Evidence Margin!
        # rel > 250 captures confident PubChem outliers that popularity alone misses
        # v49: rel promotion threshold 150.0 (seyitkaangunes/casmi26-e006-pubchem-promo-rel-150-lb-0-446)
        if (S is not None and float(S) > S_TAU and pop_top >= POP_TAU) or (rel > 150.0):
"""
GATE_NEW = """        old_S = p.get('S'); old_pop = float(p.get('top_pop') or 0.0)
        old_promote = bool((old_S is not None and float(old_S) > S_TAU and old_pop >= POP_TAU) or rel > 150.0)
        S, pop_top, candidate_rel = candidate_promotion_evidence(p)
        new_promote = bool((S > S_TAU and pop_top >= POP_TAU) or candidate_rel > 150.0)
        promotion_audit.append(dict(molecule_id=mid, leader_key=p['pc_keys'][0],
            leader_smiles=p['pc'][0], original_S=old_S, original_pop=old_pop,
            channel_rel=rel, candidate_S=S, candidate_pop=pop_top, candidate_rel=candidate_rel,
            old_promote=old_promote, new_promote=new_promote, changed=old_promote != new_promote))
        # Only promotion uses candidate-specific evidence; interleaving slots retain channel rel.
        if new_promote:
"""
STATS_OLD = "print('merge stats', stats)\n"
STATS_NEW = STATS_OLD + ("pd.DataFrame(promotion_audit).to_csv('pubchem_promotion_audit.csv', index=False)\n"
                         "print('CANDIDATE PROMOTION:', len(promotion_audit), 'eligible rows;',\n"
                         "      sum(r['changed'] for r in promotion_audit), 'decisions changed')\n")


def once(text, old, new, what):
    assert text.count(old) == 1, (what, text.count(old))
    return text.replace(old, new)


def patch_literal(src, name, edit):
    """Replace the string literal assigned to `name` in cell source `src` by repr(edit(value))."""
    tree = ast.parse(src)
    hits = [n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', None) == name]
    assert len(hits) == 1, name
    seg = ast.get_source_segment(src, hits[0].value)
    assert src.count(seg) == 1, name
    return src.replace(seg, repr(edit(ast.literal_eval(seg))))


def main(src_dir, out_dir, slug, title):
    nb_path = glob.glob(os.path.join(src_dir, '*.ipynb'))[0]
    nb = json.load(open(nb_path))
    cells = nb['cells']
    i_pc = [i for i, c in enumerate(cells) if 'V17_PATCH = ' in ''.join(c['source'])]
    i_merge = [i for i, c in enumerate(cells) if 'def promote(base, base_keys' in ''.join(c['source'])]
    assert len(i_pc) == 1 and len(i_merge) == 1, (i_pc, i_merge)

    def patch(v):
        v = once(v, LIST_OLD, LIST_OLD.replace('    out, ', '    promotion_evidence = {}\n    out, '), 'evidence init')
        v = once(v, PATCH_OLD, PATCH_NEW, 'evidence store')
        return once(v, LISTED_OLD, LISTED_OLD + "    diag['promotion_evidence'] = promotion_evidence\n", 'diag')

    s = ''.join(cells[i_pc[0]]['source'])
    s = patch_literal(s, 'V17_PATCH', patch)
    s = patch_literal(s, 'RUNNER', lambda v: once(v, RUNNER_OLD, RUNNER_NEW, 'runner'))
    cells[i_pc[0]]['source'] = s.splitlines(True)

    m = ''.join(cells[i_merge[0]]['source'])
    m = once(m, LOOP_OLD, GATE_FN + LOOP_OLD, 'gate fn')
    m = once(m, GATE_OLD, GATE_NEW, 'gate')
    m = once(m, STATS_OLD, STATS_NEW, 'audit')
    cells[i_merge[0]]['source'] = m.splitlines(True)
    for c in cells:
        if c['cell_type'] == 'code':
            c['outputs'], c['execution_count'] = [], None

    os.makedirs(out_dir, exist_ok=True)
    json.dump(nb, open(os.path.join(out_dir, os.path.basename(nb_path)), 'w'), indent=1)
    meta = json.load(open(os.path.join(src_dir, 'kernel-metadata.json')))
    meta.update(id=slug, title=title, is_private=True)
    json.dump(meta, open(os.path.join(out_dir, 'kernel-metadata.json'), 'w'), indent=1)
    print('patched cells', i_pc[0], i_merge[0], '->', out_dir)


if __name__ == '__main__':
    main(*sys.argv[1:5])
