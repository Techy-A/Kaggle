"""20: free-tier generation of Gemini Flash sibling triplets (gemini-3.6/3.7/3.8-flash) on new, competition-style prompts.

What: for each prompt (topic x form x target length, built by make_prompts) asks all three Flash models through the
Google AI Studio API (the user's free-tier key; the request header comes from $Value set in the environment), keeps
only responses whose returned modelVersion equals the requested model, writes one JSON line per response to
gen/gemini_trio.jsonl (prompt_id, prompt, topic, model, model_version, text, n_words, finish_reason, usage, ts).
Only finishReason STOP responses are kept (maxOutputTokens 8192: thinking tokens count against it). Resumable (skips prompt x model pairs already written), backs off on 429/503, stops at --max-calls or --budget-min.
Why: gemini-3.7-flash is the rarest label in Open Pool v15 (3,541 rows vs 22,508 for 3.8-flash; 0 from the Benchmarks
proxy), its OOF recall is 0.427 and 176 of 450 trio passages are within-trio errors. Same-prompt triplets carry
exactly the sibling signal. Host rule (752084): self-generated data within reasonable cost is allowed, need not be
public; free tier = $0. Topics exclude the train topics (checked later by similarity to train passages).
Usage: python 20_gen_gemini_trio.py [--max-calls 30] [--budget-min 10] [--rpm 8]
"""
import argparse, hashlib, json, os, random, re, time, urllib.request, urllib.error
from pathlib import Path
HERE = Path(__file__).parent
OUT = HERE / 'gen' / 'gemini_trio.jsonl'
MODELS = ['gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash']
FORMS = ['Write a lyrical prose passage', 'Write a vivid, evocative paragraph', 'Write a short meditative reflection',
         'Write a poetic but precise explanation', 'Write a richly descriptive passage', 'Write a contemplative piece of prose']
LENGTHS = [60, 80, 100, 120, 150, 180, 220]


def topics():
    return [l.strip() for l in open(HERE / 'gen' / 'topics.txt') if l.strip() and not l.startswith('#')]


def make_prompts(seed=7):
    rng = random.Random(seed); out = []
    for t in topics():
        f = rng.choice(FORMS); n = rng.choice(LENGTHS)
        p = f'{f} of about {n} words on {t}. Use plain prose only: no title, no headings, no lists, no markdown.'
        out.append(dict(prompt_id='g' + hashlib.md5(p.encode()).hexdigest()[:10], topic=t, prompt=p))
    return out


def call(model, prompt, key, timeout=180):
    body = json.dumps({'contents': [{'parts': [{'text': prompt}]}], 'generationConfig': {'maxOutputTokens': 8192}}).encode()
    req = urllib.request.Request(f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent', data=body,
                                 headers={'x-goog-api-key': key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--max-calls', type=int, default=30); ap.add_argument('--budget-min', type=float, default=10)
    ap.add_argument('--rpm', type=float, default=8); a = ap.parse_args()
    key = os.environ.get('Value', ''); assert key, 'API key env var not set'
    OUT.parent.mkdir(exist_ok=True)
    done = set()
    if OUT.exists():
        for l in open(OUT):
            r = json.loads(l); done.add((r['prompt_id'], r['model']))
    t0, calls, stats = time.time(), 0, {'ok': 0, 'mismatch': 0, 'err429': 0, 'err503': 0, 'other': 0, 'empty': 0}
    gap = 60.0 / a.rpm
    for P in make_prompts():
        for m in MODELS:
            if (P['prompt_id'], m) in done: continue
            if calls >= a.max_calls or time.time() - t0 > a.budget_min * 60:
                print(json.dumps(dict(calls=calls, minutes=round((time.time() - t0) / 60, 1), **stats))); return
            for attempt in range(4):
                calls += 1; ts = time.time()
                try:
                    d = call(m, P['prompt'], key)
                    mv = d.get('modelVersion', ''); c = (d.get('candidates') or [{}])[0]
                    txt = ''.join(x.get('text', '') for x in (c.get('content') or {}).get('parts', []) if not x.get('thought')).strip()
                    if not txt: stats['empty'] += 1; break
                    if c.get('finishReason') != 'STOP': stats.setdefault('truncated', 0); stats['truncated'] += 1; break   # thinking tokens can exhaust the budget
                    if mv and not mv.startswith(m): stats['mismatch'] += 1; print('MODEL MISMATCH', m, mv); break
                    rec = dict(P, model=m, model_version=mv, text=txt, n_words=len(txt.split()), finish_reason=c.get('finishReason'),
                               usage=d.get('usageMetadata'), ts=round(ts))
                    with open(OUT, 'a') as f: f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    stats['ok'] += 1; break
                except urllib.error.HTTPError as e:
                    k = 'err429' if e.code == 429 else 'err503' if e.code == 503 else 'other'; stats[k] += 1
                    msg = e.read()[:300].decode('utf-8', 'ignore')
                    if e.code == 429 and ('per day' in msg.lower() or 'PerDay' in msg):
                        print('DAILY QUOTA', m, msg[:200]); print(json.dumps(dict(calls=calls, **stats))); return
                    time.sleep(min(120, 15 * (attempt + 1)))
                except Exception as e:
                    stats['other'] += 1; time.sleep(10)
                finally:
                    time.sleep(max(0.0, gap - (time.time() - ts)))
    print(json.dumps(dict(calls=calls, minutes=round((time.time() - t0) / 60, 1), **stats, finished=True)))


if __name__ == '__main__':
    main()
