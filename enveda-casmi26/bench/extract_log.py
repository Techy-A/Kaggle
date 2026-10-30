"""Turn a Kaggle kernel log (JSON lines of {stream_name, time, data}) into plain text; print BENCH_JSON payloads
and, if present, decode the BENCH_BLOB lines (bench477) into bench477_intermediates.tar.gz next to the JSON output."""
import json, sys
raw = open(sys.argv[1]).read().strip()
start = raw.find('[')
try:
    items = json.loads(raw[start:])
except Exception:
    items = []
    for line in raw[start:].splitlines():
        line = line.strip().lstrip('[,').rstrip(']')
        try: items.append(json.loads(line))
        except Exception: pass
text = ''.join(i.get('data', '') for i in items)
open(sys.argv[2], 'w').write(text)
res = {}
for l in text.splitlines():
    if l.startswith('BENCH_JSON '):
        res.update(json.loads(l[len('BENCH_JSON '):]))
json.dump(res, open(sys.argv[3], 'w'))
print('log lines', text.count('\n'), '| result blocks', list(res))
blob = ''.join(l[len('BENCH_BLOB '):] for l in text.splitlines() if l.startswith('BENCH_BLOB '))
if blob:   # bench477: the intermediates tarball printed into the log
    import base64, os
    out = os.path.join(os.path.dirname(os.path.abspath(sys.argv[3])), 'bench477_intermediates.tar.gz')
    open(out, 'wb').write(base64.b64decode(blob)); print('wrote', out)
